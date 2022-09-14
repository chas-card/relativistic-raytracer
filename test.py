import lib.World as W
import numpy as np
from stl import mesh

from time import process_time

FARAWAY = 1.0e+39

source = np.array((0, 0.35, -1))

w, h = (200, 200)
r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1, 1 / r + .25, 1, -1 / r + .25)
x = np.tile(np.linspace(S[0], S[2], w), h)
y = np.repeat(np.linspace(S[1], S[3], h), w)

coords = np.stack((x, y, np.zeros(x.shape[0])), axis=0)
dirs = coords - source[:, np.newaxis]

m = mesh.Mesh.from_file('models/block100.stl')
direction = dirs

frame = W.Frame((0, 0, 0))
obj = W.MeshObject([0, 0, 0], frame, m, (0,))

print(direction.shape)
print(np.repeat(source[:, np.newaxis], direction.shape[1], axis=1).shape)

t1s = process_time()
arr1, n1 = obj.intersect_old(np.repeat(source[:, np.newaxis], direction.shape[1], axis=1), direction)
t1e = process_time()

t2s = process_time()
arr2, n2 = obj.intersect(np.repeat(source[:, np.newaxis], direction.shape[1], axis=1), direction)
t2e = process_time()

print(np.max(np.abs(arr2 - arr1)))
print(np.max(np.abs(n2 - n1)))
print(np.max(np.abs(n1)))
print("time1: " + str(t1e-t1s))
print("time2: " + str(t2e-t2s))
