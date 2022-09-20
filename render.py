from lib import World as W
import numpy as np
from stl import mesh

# testing

f0=W.Frame([0,0,0])
f1=W.Frame([-0.1*W.c,0,0],f0)
f3=W.Frame([-0.8*W.c,0,0],f0)

cam1 = W.Camera(0,(-80,50,-100),(np.pi/12,-np.pi/6),10,f0,3)

b1 = W.MeshObject(np.array((-5,5-120,120)),f0,mesh.Mesh.from_file('models/block100.stl'),np.array((0,1,0)), mirror=0.1)
b2 = W.MeshObject(np.array((-5-570,5-120,60)),f3,mesh.Mesh.from_file('models/block100.stl'),np.array((.8,.5,0)), mirror=0.1)
b3 = W.MeshObject(np.array((-5-120,5-40,220)),f1,mesh.Mesh.from_file('models/ico50.stl'),np.array((.4,0,0.8)), mirror=0.1)
scene1 = W.Scene(cam1, np.array((-1000, 2000, -1000)), [b1, b2, b3])

scene1.render().show()
