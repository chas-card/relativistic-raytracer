from lib import World as W
import numpy as np
from stl import mesh
from tqdm import tqdm

t0 = 0
t1 = 10

f0 = W.Frame([0,0,0])
f1 = W.Frame([-0.5*W.c,0,0],f0)

camera = W.Camera(0,(0,100,-1200),(np.pi/12,-np.pi/6),10,f0,2)

numobjs = 11
offset = 1000
objpos = (np.array([1, 0, 0], dtype=W.np_type)[:,np.newaxis] * np.linspace(-3000+offset, 3000+offset, numobjs, dtype=W.np_type)).T
objpos[:, 1] += -300
print(objpos)
objects = [W.MeshObject(objpos[i], f1, mesh.Mesh.from_file('models/cube_indent_100.stl'), np.array((0,1,0))) for i in range(numobjs)]

scene = W.Scene(camera, np.array((3000, 10000, -3000)), objects)

for t in tqdm(range(t0, t1)):
    camera.set_time()
    scene.render().show()