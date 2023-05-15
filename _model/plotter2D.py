import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import os

def plotTrajectory2D( simId, polarization, momentum, locations, N, D):
    fig, axs = plt.subplots(1, D, gridspec_kw={'width_ratios': [1, 1]}) #, 'height_ratios': [1]})
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

    figname = f'traj{simId}_2d.pdf'
    print(f"saving figure {figname}..")
    plt.savefig(figname)
    print(f"done!")

def plotSwarm2DFinal(simId, tidx, locations, directions, followcenter=False, dynamicscope=True):
    fig = plt.figure()
    ax = plt.axes()
    #fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300, projection='3d')
    #ax = fig.add_subplot(111, projection='3d')
    locations = locations[tidx,:,:]
    directions = directions[tidx,:,:]
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

    ax.quiver(locations[:,0],locations[:,1], directions[:,0], directions[:,1], color=colors)
    
    fig.tight_layout()
    figname = f'swarm{simId}_{tidx}_2d.pdf'
    print(f"saving figure {figname}..")
    plt.savefig(figname, dpi=400)
    plt.close('all')

def plotSwarm2D( sim, t, followcenter, step, numTimeSteps, dynamicscope=True):
    fig = plt.figure()
    if (step > numTimeSteps - 3):
        fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15), dpi=300)
    else:
        fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15))
    _.set_visible(False)
    ax = fig.add_subplot(211)
    locations = []
    directions = []
    history = []
    for fish in sim.fishes:
        locations.append(fish.location)
        directions.append(fish.curDirection)
        history.append(fish.history)
    locations = np.array(locations)
    directions = np.array(directions)
    history = np.array(history)

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

    ax.quiver(np.array(locations[:,0]),np.array(locations[:,1]),
            np.array(directions[:,0]), np.array(directions[:,1]),
            color=colors)
    ax.set_aspect('equal')
#ax.plot(history[:,:,0] , history[:,:,1])
    displ = 30
    if (followcenter):
        center = sim.computeCenter()
        if (dynamicscope):
            avgdist = sim.computeAvgDistCenter(center)
            displx = avgdist/2.
            ax.set_xlim([center[0]-displx-displ,center[0]+displx+displ])
            ax.set_ylim([center[1]-displx-displ,center[1]+displx+displ])
        else:
            ax.set_xlim([center[0]-displ,center[0]+displ])
            ax.set_ylim([center[1]-displ,center[1]+displ])
    if (sim.plotShortestDistance):
        for fish in sim.fishes:
            ax2.plot(np.array(fish.distanceToNearestNeighbour), alpha=0.5)
        ax2.set_xlim([0, numTimeSteps])
        ax2.axhline(sim.rRepulsion, linestyle='--')
        ax2.set_ylim([0.,10.])
    else:
        x  = np.arange(0, len(sim.angularMoments))
		#if step == 1:
		#	exit(0)
        ax2.plot(x, np.array(sim.angularMoments), '-b', label='Angular Moment')
        ax2.plot(x, np.array(sim.polarizations), '-r', label='Polarization')
        ax2.set_xlim([0, numTimeSteps])
        ax2.set_ylim([0.,1.])
    
        savestringname = "_figures/swarm_t={:04d}_2D".format(t)
        nameexists = True
        iternumber = 0
        strname = savestringname
        # while nameexists:
        #     strname = savestringname + "_{:04n}_".format(iternumber) + ".png"
        #     nameexists = os.path.exists(strname)
        #     iternumber += 1 
    plt.savefig(strname)
    print(strname)
    plt.close('all')


def finalplotSwarm2D( sim, t, followcenter, step, numTimeSteps, dynamicscope=True):
	if (step == numTimeSteps - 1):
		x  = np.arange(0, step+1)
		plt.figure(figsize=(452.0 / 72.27, 452.0*(5**.5 - 1) / 2 / 72.27), dpi=300)
		plt.plot(x, np.array(sim.angularMoments), '-b', label='Angular Moment')
		plt.plot(x, np.array(sim.polarizations), '-r', label='Polarization')
		plt.xlim([0, numTimeSteps])
		plt.ylim([0.,1.])
		#ax2.legend(frameon=False, loc='upper center', ncol=2)
		plt.savefig("_figures/2Dswarm_t={:04d}_2D.png".format(t))
		plt.close('all')


def plotSwarmSphere( sim, t, i ):
	fig = plt.figure()
	locations = []
	directions = []
	for fish in sim.swarm:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],
		      directions[:,0], directions[:,1])
	# Create a sphere
	r = 1
	pi = np.pi
	cos = np.cos
	sin = np.sin
	phi = np.mgrid[0.0:pi:100j]
	x = r*cos(phi)
	y = r*sin(phi)
	ax.plot_surface(x, y,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
	ax.set_aspect('equal', 'box')
	#ax.set_xlim([-2,2])
	#ax.set_ylim([-2,2])
	plt.savefig("_figures/swarm_t={}_sphere_i={}.png".format(t,i))
	plt.close('all')

def plotFishs( fishs, i, t, type ):
	if fishs.size == 0:
		print("no fish of type {}".format(type))
		return
	fig = plt.figure()
	locations = []
	directions = []
	for fish in fishs:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],
		      directions[:,0], directions[:,1])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	plt.savefig("_figures/{}_t={}_i={}.png".format(type, t, i))
	plt.close()

def plotFish( fish, i, t ):
	fig = plt.figure()
	loc = fish.location
	vec = fish.curDirection
	ax.quiver(loc[0], loc[1], vec[0], vec[1])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	plt.savefig("_figures/fish_t={}_i={}.png".format(t, i))
	plt.close()

def plotRot( vec1, vec2, rotvec, angle ):
	fig = plt.figure()
	locations = [vec1,vec2,rotvec]
	vecs = np.array([vec1,vec2,rotvec])
	loc = np.zeros(2)
	ax.quiver(loc, loc, vecs[:,0], vecs[:,1], color=['green','red'])
	ax.set_title("rotation by {} degree".format(angle))
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	plt.show()
