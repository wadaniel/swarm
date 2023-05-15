#!/usr/bin/env python3

import sys
sys.path.append('./_model')

import re
import json
import argparse
import numpy as np
from scipy.signal import savgol_filter

from plotter2D import plotSwarm2DFinal
from plotter3D import plotSwarm3DFinal

from swarm import *

parser = argparse.ArgumentParser()
parser.add_argument('--resdir', help='Directory of IRL results.', required=True)
parser.add_argument('--tfile', help='Trajectory file.', required=True)
parser.add_argument('--N', help='Num fish.', required=True, type=int)
parser.add_argument('--NN', help='Num nearest neighbours.', required=True, type=int)
parser.add_argument('--D', help='Num dimensions.', required=True, type=int)
parser.add_argument('--tidx', help='Index in trajectory.', required=True, type=int)
parser.add_argument('--nfish', help='Number of fish to plot.', required=True, type=int)
parser.add_argument('--rnn', help='Size RNN', required=True, type=int)

args = parser.parse_args()
print(args)

tridx = re.findall(r'\d+', args.tfile)
print(tridx)
assert(len(tridx) == 1)
tridx = tridx[0]

tidx = args.tidx
nfish = args.nfish

outfile=f"rewards_{tridx}_{tidx}.npz"
rfile=f"rewards_{tridx}_{tidx}.png"

trajectory = np.load(args.tfile)
locations = trajectory["locationHistory"]
directions = trajectory["directionHistory"]
print(locations.shape)
print(directions.shape)

if args.D == 2:
    plotSwarm2DFinal(tridx, tidx, np.array(locations), np.array(directions))

elif args.D == 3:
    plotSwarm3DFinal(tridx, tidx, np.array(locations), np.array(directions))

sim = swarm( N=args.N, numNN=args.NN,
    numdimensions=args.D, initType=1, movementType=2)

sim.initFromLocationsAndDirections(locations[tidx,:,:], directions[tidx,:,:])

batchSize = 1000
#rotations = np.linspace(-8/360*np.pi, +8/360*np.pi, batchSize)
rotations = np.linspace(-np.pi, +np.pi, batchSize)

states = []
for fish in range(nfish):
    for _, rot in enumerate(rotations):
        sim.fishes[fish].curDirection = sim.fishes[fish].applyrotation(sim.fishes[fish].curDirection, rot)
        sim.preComputeStates()
        states.append([sim.getState(fish).tolist()])
        sim.fishes[fish].curDirection = \
            sim.fishes[fish].applyrotation(sim.fishes[fish].curDirection, -rot)

resfile = f'{args.resdir}/latest'
with open(resfile, 'r') as infile:
    results = json.load(infile)
    actionDim = results["Problem"]["Action Vector Size"]
    featureDim = results["Problem"]["Feature Vector Size"]
    stateDim = results["Problem"]["State Vector Size"]
    rewardHp = results["Solver"]["Training"]["Best Reward Params"]["Hyperparameters"]
    policyHp = results["Solver"]["Training"]["Best Policies"]["Hyperparameters"]

import korali
k = korali.Engine()
e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = 1
e["Problem"]["Testing Batch Size"] = batchSize*nfish

e["Problem"]["Input"]["Data"] = states
e["Problem"]["Input"]["Size"] = featureDim
e["Problem"]["Solution"]["Size"] = 1

### Using a neural network solver (deep learning) for training

e["Solver"]["Type"] = "DeepSupervisor"
e["Solver"]["Mode"] = "Testing"
e["Solver"]["Loss Function"] = "Direct Gradient"
e["Solver"]["Learning Rate"] = 1e-4

e["Solver"]["Hyperparameters"] = rewardHp

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Output Weights Scaling"] = 0.001

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.rnn

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.rnn

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0
e["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = -0.5
e["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Sigmoid"


### Configuring output

e["Random Seed"] = 0xC0FFEE
e["File Output"]["Enabled"] = True
e["File Output"]["Use Multiple Files"] = True
e["Console Output"]["Frequency"] = 1
e["File Output"]["Path"] = '_korali_result_reward_evaluation'

e["Solver"]["Termination Criteria"]["Max Generations"] = 100
k.run(e)

rewards = np.array(e["Solver"]["Evaluation"]).reshape((nfish,batchSize)).T
muReward = np.mean(rewards,axis=1)
muRewardHat = savgol_filter(muReward, 49, 3)
sdevReward = np.std(rewards,axis=1)
sdevRewardHat = savgol_filter(sdevReward, 49, 3)   
#print(f"Writing reward {rewards.shape}")
#np.savez(outfile, rotations=rotations, states=np.array(states), rewards=rewards)

print(f"Plotting file {rfile}")
plt.plot(rotations, muRewardHat, '-', color='darkorange', linewidth=2)
plt.fill_between(rotations, muRewardHat+sdevRewardHat, muRewardHat-sdevRewardHat, \
    color='darkorange', alpha=0.2)

#plt.plot(rotations, rewards, '--')
plt.tight_layout()
plt.savefig(rfile)
