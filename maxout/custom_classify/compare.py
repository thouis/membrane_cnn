import h5py
import sys
import numpy as np


p1 = h5py.File(sys.argv[1], 'r')['/probabilities'][...]
p2 = h5py.File(sys.argv[2], 'r')['/probabilities'][...]

print p1.shape, p2.shape
if p1.shape != p2.shape:
    # assume p1 is gpu
    p1 = p1.transpose((0, 3, 1, 2))

print p1.shape, p2.shape
print abs(p1 - p2).max()
print np.median(abs(p1 - p2).ravel())
