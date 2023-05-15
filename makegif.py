import imageio

frames = []

time = 1000
for t in range(time):
    print(f"reading t={t}")
    fname = "_figures/swarm_t={:04d}_3D.png".format(t)
    image = imageio.imread(fname)
    frames.append(image)


imageio.mimsave('./swarm.gif', # output gif
                frames,          # array of input frames
                fps = 15)         # optional: frames per second
