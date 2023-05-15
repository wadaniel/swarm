#!/usr/bin/env python3
import sys
sys.path.append('./_model')

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from swarm import *

parser = argparse.ArgumentParser()
parser.add_argument('--rewardfile', help='Input file.', required=True)

args = parser.parse_args()
print(args)

infile = np.load(args.rewardfile)
rewards = infile["rewards"]
rotations = infile["rotations"]
print(rewards.shape)
print(rotations.shape)

plt.plot(rotations, rewards)
plt.savefig('rewards.png')
