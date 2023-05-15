import sys
import math
import json
import argparse
import numpy as np
sys.path.append('_model')
from environment import *

### Parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument('--visualize', help='whether to plot the swarm or not, default is 0', required=False, type=int, default=0)
parser.add_argument('--N', help='Swarm size.', required=False, type=int, default=10)
parser.add_argument('--NT', help='Number of steps', required=False, type=int, default=500)
parser.add_argument('--NN', help='Number of nearest neighbours', required=False, type=int, default=6)
parser.add_argument('--NL', help='Number of nodes in hidden layer', required=False, type=int, default=128)
parser.add_argument('--reward', help='Reward type (local / global)', required=False, type=str, default="global")
parser.add_argument('--exp', help='Number of experiences.', required=False, type=int, default=1000000)
parser.add_argument('--dim', help='Dimensions.', required=False, type=int, default=2)
parser.add_argument('--dat', help='Number of observed trajectories used.', type=int, required=False, default=-1)
parser.add_argument('--run', help='Run tag.', required=False, type=int, default=0)

# IRL params
parser.add_argument('--rnn', help='Reward Neural Net size.', required=False, default=8, type=int)
parser.add_argument('--ebru', help='Experiences between reward update.', required=False, default=500, type=int)
parser.add_argument('--dbs', help='Demonstration Batch Size.', required=False, default=2, type=int)
parser.add_argument('--bbs', help='Background Batch Size.', required=False, default=16, type=int)
parser.add_argument('--bss', help='Background Sample Size.', required=False, default=100, type=int)
parser.add_argument('--pol', help='Demonstration Policy (Constant, Linear or Quadratic).', required=False, default="Linear", type=str)

args = parser.parse_args()
print(args)

### check arguments
numIndividuals          = args.N
numTimesteps            = args.NT
numNearestNeighbours    = args.NN
numNodesLayer           = args.NL
exp                     = args.exp
dim                     = args.dim
run                     = args.run
visualize               = args.visualize

ndata = args.dat

assert (numIndividuals > 0) 
assert (numTimesteps > 0) 
assert (exp > 0) 
assert (numNearestNeighbours > 0) 
assert (numIndividuals > numNearestNeighbours)

# Define max Angle of rotation during a timestep
maxAngle=swarm.maxAngle


# Load data
#fname = f'_trajectories/observations_simple_{numIndividuals}_{numNearestNeighbours}_{dim}d.json'
fname = f'_trajectories/observations_extended_{numIndividuals}_{numNearestNeighbours}_{ndata}_{dim}d.json'
obsstates = []
obsactions = []
obsfeatures = []
with open(fname, 'r') as infile:
    obsjson = json.load(infile)
    obsstates = obsjson["States"]
    obsactions = obsjson["Actions"]
    obsfeatures = obsstates.copy()

print("Total observed trajectories: {}/{}/{}".format(len(obsstates), len(obsfeatures), len(obsactions)))
print(len(obsstates[0][0][0]))
print(len(obsactions[0][0][0]))
print(len(obsfeatures[0][0][0]))

### Define Korali Problem
import korali
k = korali.Engine()
e = korali.Experiment()

### Define results folder and loading previous results, if any
resultFolder = f'_result_vracer_irl_{run}/'
found = e.loadState(resultFolder + '/latest')

### IRL variables
e["Problem"]["Observations"]["States"] = obsstates[:args.dat]
e["Problem"]["Observations"]["Actions"] = obsactions[:args.dat]
e["Problem"]["Observations"]["Features"] = obsfeatures[:args.dat]
e["Problem"]["Custom Settings"]["Store Good Episodes"] = "True"

### Define Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : environment( args, x )
e["Problem"]["Agents Per Environment"] = numIndividuals
e["Problem"]["Testing Frequency"] = 50
e["Problem"]["Policy Testing Episodes"] = 10

### Define Agent Configuration 
e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Testing" if visualize else "Training"
e["Solver"]["Testing"]["Sample Ids"] = [1341, 1342, 1343, 1344, 1345]
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 128

numStates = 3*numNearestNeighbours if dim == 2 else 5*numNearestNeighbours
#numStates = dim*numNearestNeighbours
#numStates = numNearestNeighbours*5
# States (distance and angle to nearest neighbours)
for i in range(numStates):
  e["Variables"][i]["Name"] = "State " + str(i)
  e["Variables"][i]["Type"] = "State"

# Direction update left/right
e["Variables"][numStates]["Name"] = "Phi"
e["Variables"][numStates]["Type"] = "Action"
e["Variables"][numStates]["Lower Bound"] = -maxAngle
e["Variables"][numStates]["Upper Bound"] = +maxAngle
e["Variables"][numStates]["Initial Exploration Noise"] = maxAngle/2.

# Direction update up/down
if dim == 3:
    e["Variables"][numStates+1]["Name"] = "Theta"
    e["Variables"][numStates+1]["Type"] = "Action"
    e["Variables"][numStates+1]["Lower Bound"] = -maxAngle
    e["Variables"][numStates+1]["Upper Bound"] = +maxAngle
    e["Variables"][numStates+1]["Initial Exploration Noise"] = maxAngle/2.

### Set Experience Replay, REFER and policy settings
e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Feature Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False

### IRL related configuration

e["Solver"]["Optimize Max Entropy Objective"] = True
e["Solver"]["Use Fusion Distribution"] = True
e["Solver"]["Demonstration Policy"] = args.pol
e["Solver"]["Demonstration Batch Size"] = args.dbs
e["Solver"]["Background Batch Size"] = args.bbs
e["Solver"]["Background Sample Size"] = args.bss
e["Solver"]["Experiences Between Reward Updates"] = args.ebru

## Reward Function Specification

e["Solver"]["Reward Function"]["Batch Size"] = 32
e["Solver"]["Reward Function"]["Learning Rate"] = 1e-4 / numIndividuals

e["Solver"]["Reward Function"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["Reward Function"]["L2 Regularization"]["Importance"] = 0.

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.rnn

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.rnn

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Configure the neural network and its hidden layers
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 0.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = numNodesLayer

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = numNodesLayer

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Termination Criteria"]["Max Experiences"] = exp
e["Solver"]["Termination Criteria"]["Max Running Time"] = 84000
e["Solver"]["Experience Replay"]["Serialize"] = False

### Set file output configuration
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 50
e["File Output"]["Path"] = resultFolder

### Run Experiment
k.run(e)
print(args)
