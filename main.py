import argparse
import sys
sys.path.append('_model')
import json
from swarm import *
from plotter3D import *
import imageio
import math
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--record', help='whether to write states and actions to json file', action="store_true")
    parser.add_argument('--visualize', help='whether to plot the swarm or not', action="store_true")
    parser.add_argument('--N', help='number of fish', required=False, type=int, default=10)
    parser.add_argument('--NT', help='number of timesteps to simulate', required=False, type=int, default=500)
    parser.add_argument('--NN', help='number of nearest neighbours used for state/reward', required=False, type=int, default=3)
    parser.add_argument('--D', help='number of dimensions of the simulation', required=False, type=int, default=2)
    parser.add_argument('--centered', help='if plotting should the camera be centered or not', required=False, type=int, default=1)
    parser.add_argument('--initialization', help='how the fishes should be initialized. 0 for grid, 1 for on circle or sphere, 2 for within a circle or a sphere', required=False, type=int, default=1)
    parser.add_argument('--psi', help='gives the initial polarization of the fish', required=False, type=float, default=-1.)
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=10)
    
    args = vars(parser.parse_args())

    numIndividuals       = args["N"]
    numTimeSteps         = args["NT"]
    numNearestNeighbours = args["NN"]
    numdimensions        = args["D"]
    followcenter         = args["centered"]
    initializationType   = args["initialization"]
    psi                  = args["psi"]
    seed                 = args["seed"]

    assert numIndividuals > numNearestNeighbours, print("numIndividuals must be bigger than numNearestNeighbours")

    print(numdimensions)
    sim  = swarm( numIndividuals, numNearestNeighbours,  numdimensions, 2, initializationType, _psi=psi)
    step = 0
    done = False
    action = np.zeros(shape=(sim.dim), dtype=float)

    obsstates = []
    obsactions = []
    obsrewards = []
    fname = f'_trajectories/observations_simple_{numIndividuals}_{numNearestNeighbours}.json'
    observations = {}

    if args['record']:
        Path("./_trajectories").mkdir(parents=True, exist_ok=True)
        try:
            f = open(fname)
            observations = json.load(f)
            obsstates = observations["States"]
            obsactions = observations["Actions"]
            obsrewards = observations["Rewards"]
            print(f'{len(obsstates)} trajectories loaded')
        except:
            print(f'File {fname} not found, init empty obs file')
    
    states = []
    actions = []
    rewards = []

    while (step < numTimeSteps):
        print("timestep {}/{}".format(step+1, numTimeSteps))
        # if enable, plot current configuration
        if args["visualize"]:
            Path("./_figures").mkdir(parents=True, exist_ok=True)
            if(sim.dim == 3):
                #finalplotSwarm3D( sim, step, followcenter, step, numTimeSteps)
                plotSwarm3D( sim, step, followcenter, step, numTimeSteps)
            else:
                plotSwarm2D( sim, step, followcenter, step, numTimeSteps)
        
        # compute pair-wise distances and view-angles
        done = sim.preComputeStates()

        sim.angularMoments.append(sim.computeAngularMom())
        sim.polarizations.append(sim.computePolarisation())
        sim.move_calc()
         
        if args["record"]:
            state = [ list(sim.getState(i)) for i in range(numIndividuals) ]
            action = [ list(sim.fishes[i].getAction()) for i in range(numIndividuals) ]
            reward = [ list(sim.getGlobalReward()) ]

            print(f"rew {reward[0][0]} {sim.dim}d (record)")
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            for i in np.arange(sim.N):
                sim.fishes[i].curDirection = sim.fishes[i].applyrotation(sim.fishes[i].curDirection, action[i])
                sim.fishes[i].updateLocation()

        else:
            reward = [ list(sim.getGlobalReward()) ]
            print(f"rew {reward[0][0]} {sim.dim}d")
            
            # update swimming directions
            for i in np.arange(sim.N):
                state  = sim.getState(i)
                sim.fishes[i].updateDirection()
                sim.fishes[i].updateLocation()


        step += 1
    
    obsstates.append(states)
    obsactions.append(actions)
    obsrewards.append(rewards)

    if args["record"]:
        observations["States"] = obsstates
        observations["Actions"] = obsactions
        observations["Rewards"] = obsrewards
        with open(fname,'w') as f:
            json.dump(observations, f)

        print(f"Saved {len(obsstates)} trajectories")
