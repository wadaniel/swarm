import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import imageio


def plotTrajectory3D( simId, polarization, momentum, locations, N, D):
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

    figname = f'traj{simId}.pdf'
    print(f"saving figure {figname}..")
    plt.savefig(figname)
    print(f"done!")

def plotSwarm3DFinal(idx, tidx, locations, directions, followcenter=False, dynamicscope=True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
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

    ax.quiver(locations[:,0],locations[:,1],locations[:,2], directions[:,0], directions[:,1], directions[:,2], colors=colors, normalize=True, length=1.)
    
    fig.tight_layout()
    figname=f"swarm{idx}_{tidx}_3d.pdf"
    print(f"saving figure {figname}..")
    plt.savefig(figname, dpi=400)
    plt.close('all')

def plotSwarm3DMovie( simId, followcenter, dynamicscope, N, 
        locationHistory, directionHistory, centerHistory, avgDistHistory, angularMomentHistory, polarizationHistory ):
        Path("./_figures").mkdir(parents=True, exist_ok=True)

        frames = []
        numTimeSteps = len(locationHistory)
        print(f"plotting {numTimeSteps} figures..")
        print(locationHistory[0])
        print(centerHistory[0])
        for t in range(numTimeSteps):
            fig = plt.figure()
            if (t > numTimeSteps - 3):
                    fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15), dpi=300)
            else:
                    fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15))
            _.set_visible(False)
            ax = fig.add_subplot(211, projection='3d')
            locations = np.array(locationHistory[t])
            directions = np.array(directionHistory[t])
            
            colors = []
            norm = Normalize(vmin=0, vmax=N)
            csel = plt.cm.inferno(norm(np.arange(N)))
            for i in range(N):
                colors.append(csel[i])
            for i in range(N):
                colors.append(csel[i])
                colors.append(csel[i])

            ax.quiver(locations[:,0],locations[:,1],locations[:,2],
                          directions[:,0], directions[:,1], directions[:,2], 
                          colors=colors)

            displ = 5
            if (followcenter):
                    center = centerHistory[t]
                    if (dynamicscope):
                            avgdist = avgDistHistory[t]
                            displx = avgdist/2.
                            ax.set_xlim([center[0]-displx-displ,center[0]+displx+displ])
                            ax.set_ylim([center[1]-displx-displ,center[1]+displx+displ])
                            ax.set_zlim([center[2]-displx-displ,center[2]+displx+displ])
                    else:
                            ax.set_xlim([center[0]-displ,center[0]+displ])
                            ax.set_ylim([center[1]-displ,center[1]+displ])
                            ax.set_zlim([center[2]-displ,center[2]+displ])
            else:
                    ax.set_xlim([-displ,displ])
                    ax.set_ylim([-displ,displ])
                    ax.set_zlim([-displ,displ])

            x  = np.arange(0, t+1)
            ax2.plot(x, angularMomentHistory[:t+1], '-b', label='Angular Moment')
            ax2.plot(x, polarizationHistory[:t+1], '-r', label='Polarization')
            ax2.set_xlim([0, numTimeSteps])
            ax2.set_ylim([0.,1.])
            figName = f"_figures/swarm_{simId}_t={t:04d}_3D.png"
            #ax2.legend(frameon=False, loc='upper center', ncol=2)
            plt.savefig(figName)
            plt.close('all')

            image = imageio.imread(figName)
            frames.append(image)

        print(f"Saving _figures/swarm_{simId}.gif")
        imageio.mimsave(f'_figures/swarm_{simId}.gif',
            frames,             # array of input frames
            fps = 20)           # optional: frames per second


def plotSwarm3D( sim, t, followcenter, step, numTimeSteps, dynamicscope=True):
	fig = plt.figure()
	if (step > numTimeSteps - 3):
		fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15), dpi=400)
	else:
		fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15))
	_.set_visible(False)
	ax = fig.add_subplot(211, projection='3d')
	locations = []
	directions = []
	for fish in sim.fishes:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	cmap = cm.jet
	norm = Normalize(vmin=0, vmax=sim.N)
	ax.quiver(locations[:,0],locations[:,1],locations[:,2],
		      directions[:,0], directions[:,1], directions[:,2], 
		      color=cmap(norm(np.arange(sim.N))))
	displ = 5
	if (followcenter):
		center = sim.computeCenter()
		if (dynamicscope):
			avgdist = sim.computeAvgDistCenter(center)
			displx = avgdist/2.
			ax.set_xlim([center[0]-displx-displ,center[0]+displx+displ])
			ax.set_ylim([center[1]-displx-displ,center[1]+displx+displ])
			ax.set_zlim([center[2]-displx-displ,center[2]+displx+displ])
		else:
			ax.set_xlim([center[0]-displ,center[0]+displ])
			ax.set_ylim([center[1]-displ,center[1]+displ])
			ax.set_zlim([center[2]-displ,center[2]+displ])
	else:
		ax.set_xlim([-displ,displ])
		ax.set_ylim([-displ,displ])
		ax.set_zlim([-displ,displ])
	x  = np.arange(0, step+1)
	ax2.plot(x, np.array(sim.angularMoments), '-b', label='Angular Moment')
	ax2.plot(x, np.array(sim.polarizations), '-r', label='Polarization')
	ax2.set_xlim([0, numTimeSteps])
	ax2.set_ylim([0.,1.])
	#ax2.legend(frameon=False, loc='upper center', ncol=2)
	plt.savefig("_figures/swarm_t={:04d}_3D.png".format(t))
	plt.close('all')


def finalplotSwarm3D( sim, t, followcenter, step, numTimeSteps, dynamicscope=True):
	if (step == numTimeSteps - 1):
		x  = np.arange(0, step+1)
		plt.figure(figsize=(452.0 / 72.27, 452.0*(5**.5 - 1) / 2 / 72.27), dpi=300)
		plt.plot(x, np.array(sim.angularMoments), '-b', label='Angular Moment')
		plt.plot(x, np.array(sim.polarizations), '-r', label='Polarization')
		plt.xlim([0, numTimeSteps])
		plt.ylim([0.,1.])
		#ax2.legend(frameon=False, loc='upper center', ncol=2)
		plt.savefig("_figures/3Dswarm_t={:04d}_3D.png".format(t))
		plt.close('all')

def plotSwarmSphere( sim, t, i ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	locations = []
	directions = []
	for fish in sim.swarm:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],locations[:,2],
		      directions[:,0], directions[:,1], directions[:,2])
	# Create a sphere
	r = 1
	pi = np.pi
	cos = np.cos
	sin = np.sin
	phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
	x = r*sin(phi)*cos(theta)
	y = r*sin(phi)*sin(theta)
	z = r*cos(phi)
	ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
	ax.set_aspect('equal', 'box')
	#ax.set_xlim([-2,2])
	#ax.set_ylim([-2,2])
	#ax.set_zlim([-2,2])
	plt.savefig("_figures/swarm_t={}_sphere_i={}.png".format(t,i))
	plt.close()

def plotFishs( fishs, i, t, type ):
	if fishs.size == 0:
		print("no fish of type {}".format(type))
		return
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	locations = []
	directions = []
	for fish in fishs:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],locations[:,2],
		      directions[:,0], directions[:,1], directions[:,2])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	ax.set_zlim([-2,2])
	plt.savefig("_figures/{}_t={}_i={}.png".format(type, t, i))
	plt.close('all')

def plotFish( fish, i, t ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	loc = fish.location
	vec = fish.curDirection
	ax.quiver(loc[0], loc[1], loc[2], vec[0], vec[1], vec[2])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	ax.set_zlim([-2,2])
	plt.savefig("_figures/fish_t={}_i={}.png".format(t, i))
	plt.close()

def plotRot( vec1, vec2, rotvec, angle ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	locations = [vec1,vec2,rotvec]
	vecs = np.array([vec1,vec2,rotvec])
	loc = np.zeros(3)
	ax.quiver(loc, loc, loc, vecs[:,0], vecs[:,1], vecs[:,2], color=['green','red','black'])
	ax.set_title("rotation by {} degree".format(angle))
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	plt.show()
