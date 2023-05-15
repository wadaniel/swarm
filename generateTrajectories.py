import argparse
import sys
sys.path.append('_model')
import json
from swarm import *
from plotter3D import *
import math
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', help='number of fish', required=False, type=int, default=10)
    parser.add_argument('--NT', help='number of timesteps to simulate', required=False, type=int, default=1000)
    parser.add_argument('--NN', help='number of nearest neighbours used for state/reward', required=False, type=int, default=3)
    parser.add_argument('--D', help='number of dimensions of the simulation', required=False, type=int, default=2)
    parser.add_argument('--initialization', help='how the fishes should be initialized. 0 for grid, 1 for on circle or sphere, 2 for within a circle or a sphere', required=False, type=int, default=1)
    parser.add_argument('--psi', help='gives the initial polarization of the fish', required=False, type=float, default=-1.)
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=1)
    parser.add_argument('--num', help='number of trajectories to produce', required=False, type=int, default=1)
    parser.add_argument('--visualize', help='whether to plot the swarm or not', action="store_true")

    args = vars(parser.parse_args())

    numIndividuals       = args["N"]
    numTimeSteps         = args["NT"]
    numNearestNeighbours = args["NN"]
    numdimensions        = args["D"]
    initializationType   = args["initialization"]
    psi                  = args["psi"]
    seed                 = args["seed"]
    numTrajectories      = args["num"]

    assert numIndividuals > numNearestNeighbours, print("numIndividuals must be bigger than numNearestNeighbours")

    Path("./_trajectories").mkdir(parents=True, exist_ok=True)
    fname = f'./_trajectories/observations_extended_{numIndividuals}_{numNearestNeighbours}_{numTrajectories}_{numdimensions}d.json'
    
    observations = {}
    obsstates = []
    obsactions = []
    obsrewards = []
    obslocations = []
    obsdirections = []
    obsseeds = []
    obscumrewards = []
    obsangularmomentum = []
    obspolarization = []

    try:
        f = open(fname)
        observations = json.load(f)
        obsstates = observations["States"]
        obsactions = observations["Actions"]
        obsrewards = observations["Rewards"]
        obsseeds = observations["Seeds"]
        obslocations = observations["Locations"]
        obsdirections = observations["Directions"]
        obscumrewards = observations["Cumulative Rewards"]
        obsangularmomentum = observations["Angular Momentum"]
        obspolarization = observations["Polarization"]
        print(f'{len(obsstates)} trajectories loaded')
    except Exception as ex:
        print(f'Exception raised {ex}, init empty obs file')

    count = len(obsstates)

    while count < numTrajectories:
        sim  = swarm( numIndividuals, numNearestNeighbours,  numdimensions, 2, initializationType, _psi=psi, seed=seed+count )
        action = np.zeros(shape=(sim.dim), dtype=float)
       
        states = []
        actions = []
        rewards = []
        
        step = 0
        cumReward = 0

        centerHistory = []
        avgDistHistory = []
        locationHistory = []
        directionHistory = []

        while (step < numTimeSteps):

            centerHistory.append(sim.computeCenter())
            avgDistHistory.append(sim.computeAvgDistCenter(centerHistory[-1]))

            locations = []
            directions = []

            for i in np.arange(sim.N):
                locations.append(list(sim.fishes[i].location.copy()))
                directions.append(list(sim.fishes[i].curDirection.copy()))

            locationHistory.append(locations.copy())
            directionHistory.append(directions.copy())

            if args["visualize"]:
                Path("./_figures").mkdir(parents=True, exist_ok=True)
                followcenter = True
                if(sim.dim == 3):
                    plotSwarm3D( sim, step, followcenter, step, numTimeSteps)
                else:
                    plotSwarm2D( sim, step, followcenter, step, numTimeSteps)
 
            sim.preComputeStates()
            sim.angularMoments.append(sim.computeAngularMom())
            sim.polarizations.append(sim.computePolarisation())
            sim.move_calc()
             
            state = [ list(sim.getState(i)) for i in range(numIndividuals) ]
            action = [ list(sim.fishes[i].getAction()) for i in range(numIndividuals) ]
            reward = list(sim.getGlobalReward())[0]
            print(f"{count}: {step+1} reward (avg) {reward} ({cumReward/(step+1)})")

            if np.isnan(state).any() == True:
                print("nan state detected, abort trajectory")
                reward = -99
                break

            if np.isnan(action).any() == True:
                print("nan action detected, abort trajectory")
                reward = -99
                break
            
            for i in np.arange(sim.N):
                assert np.abs(action[i][0]) < np.pi, f"invalid action {action[i]}"
                sim.fishes[i].curDirection = sim.fishes[i].applyrotation(sim.fishes[i].curDirection, action[i])
                sim.fishes[i].updateLocation()


            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            cumReward += reward
            step += 1

            if step > 300 and reward < 0.6:
                break

        cumReward /= numTimeSteps
     

        if cumReward > 0.8:
            #plotSwarm3DEnv(count, True, True, sim.N, locationHistory, directionHistory, centerHistory, avgDistHistory, sim.angularMoments, sim.polarizations)
            obsstates.append(states)
            obsactions.append(actions)
            obsrewards.append(rewards)
            obsseeds.append(seed+count)
            obscumrewards.append(cumReward)
            obslocations.append(locationHistory)
            obsdirections.append(directionHistory)
            obspolarization.append(sim.polarizations)
            obsangularmomentum.append(sim.angularMoments)

        count = len(obsstates)

        if count % 10 == 0:
            print(f"dumping trajectories {fname}")
            observations["States"] = obsstates
            observations["Actions"] = obsactions
            observations["Rewards"] = obsrewards
            observations["Seeds"] = obsseeds
            observations["Cumulative Rewards"] = obscumrewards
            ## Addon
            observations["Locations"] = obslocations
            observations["Directions"] = obsdirections
            observations["Angular Moments"] = obsangularmomentum
            observations["Polarization"] = obspolarization

            with open(fname,'w') as f:
                json.dump(observations, f)

            print(f"Saved {len(obsstates)} trajectories")
            print(f"Seeds used {obsseeds}")
            print(f"Cumrewards {obscumrewards}")
