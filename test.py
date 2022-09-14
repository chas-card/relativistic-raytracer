import lib.World as W
import numpy as np
from stl import mesh

from time import process_time

FARAWAY = 1.0e+39

source = np.array((0, 0.35, -1))

w, h = (640, 480)
r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1, 1 / r + .25, 1, -1 / r + .25)
x = np.tile(np.linspace(S[0], S[2], w), h)
y = np.repeat(np.linspace(S[1], S[3], h), w)

coords = np.stack((x, y, np.zeros(x.shape[0])), axis=0)
dirs = coords - source[:, np.newaxis]

m = mesh.Mesh.from_file('models/monke.stl')
direction = dirs

frame = W.Frame((0, 0, 0))
obj = W.MeshObject([0, 0, -80], frame, m, (0,))

t1s = process_time()
arr1 = obj.intersect(source, direction)
t1e = process_time()

t2s = process_time()
arr2 = obj.np_intersect(source, direction)
t2e = process_time()

print(np.max(np.abs(arr2 - arr1)))
print("time1: " + str(t1e-t1s))
print("time2: " + str(t2e-t2s))
