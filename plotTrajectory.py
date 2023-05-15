import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plotSwarm3D(idx, locations, directions):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    locations = locations[-1,:,:]
    directions = directions[-1,:,:]
    N, _ = locations.shape
    
    cmap = plt.cm.inferno
    norm = Normalize(vmin=0, vmax=N)

    colors = []
    norm = Normalize(vmin=0, vmax=N)
    csel = plt.cm.inferno(norm(np.arange(N)))
    for i in range(N):
        colors.append(csel[i])
    for i in range(N):
        colors.append(csel[i])
        colors.append(csel[i])

    ax.quiver(locations[:,0],locations[:,1],locations[:,2], directions[:,0], directions[:,1], directions[:,2], colors=colors, normalize=True, length=1.)
    
    fig.tight_layout()
    plt.savefig(f"swarm{idx}.pdf", dpi=400)
    plt.close('all')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='trajectory file', required=True, type=str)
    parser.add_argument('--idx', help='traj index', required=False, type=int, default=0)
    parser.add_argument('--i', help='dim idx', required=False, type=int, default=0)
    parser.add_argument('--j', help='dim idx', required=False, type=int, default=1)

    args = parser.parse_args()
    assert(args.i != args.j)
  
    if args.file.endswith('json'):
        # Opening JSON file
        f = open(args.file)
      
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        locations = data["Locations"]
        momentum = data["Angular Moments"]
        polarization = data["Polarization"]
        directions = data["Directions"]

        locations = np.array(locations[args.idx])
        polarization = np.array(polarization[args.idx])
        momentum = np.array(momentum[args.idx])
        directions = np.array(directions[args.idx])

    else:
        f = np.load(args.file)
        locations = f["locationHistory"]
        directions = f["directionHisory"]
        plotSwarm3D(args.idx, locations, directions)
        exit()

    print(f"loaded {len(locations)} trajectories")
    NT, N, D = locations.shape

    print(f"plotting..")
    plotSwarm3D(args.idx, locations, directions)
    fig, axs = plt.subplots(1, D, gridspec_kw={'width_ratios': [1, 1, 1]}) #, 'height_ratios': [1]})
    colors = plt.cm.Oranges(np.linspace(0, 1, N))

    axs[0].plot(polarization, color='steelblue')
    axs[0].plot(momentum, color='coral')
    axs[0].set_box_aspect(1)
    axs[0].set_yticks([0.1, 0.5, 0.9])
    axs[0].set_ylim([0.0, 1.0])
    for d in range(D-1):
        for fish in range(N):
          traj = locations[:,fish, :]
          axs[1+d].plot(traj[:,d], traj[:,d+1], color=colors[fish])
        axs[1+d].set_aspect('equal') #, 'box')
        axs[1+d].set_box_aspect(1)

    fig.tight_layout()

    figname = f'traj{args.idx}.pdf'
    print(f"saving figure {figname}..")
    plt.savefig(figname)
    print(f"done!")

