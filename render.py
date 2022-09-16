from lib import World as W
import numpy as np
from stl import mesh

# testing

f0=W.Frame([0,0,0])
f3=W.Frame([-0*W.c,0,0],f0)

cam1 = W.Camera(0,(-80,50,-100),(np.pi/12,-np.pi/6),10,f0,2)

b1 = W.MeshObject(np.array((-5,5-100,50)),f0,mesh.Mesh.from_file('models/block100.stl'),np.array((0,1,0)), mirror=0.2)
b2 = W.MeshObject(np.array((-5-60,5-40,100)),f3,mesh.Mesh.from_file('models/ico50.stl'),np.array((.8,.5,0)), mirror=0.2)
scene1 = W.Scene(cam1, np.array((3000, 10000, -3000)), [b1, b2])

scene1.render().show()
