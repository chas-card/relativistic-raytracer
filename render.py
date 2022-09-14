from lib import World as W
import numpy as np
from stl import mesh

# testing

f0=W.Frame([0,0,0])
f1=W.Frame([-7.99,0,0],f0)
f2=W.Frame([.8*W.c,0,0],f0)
f3=W.Frame([-.8*W.c,0,0],f0)

cam = W.Camera(0,(-80,50,-100),(np.pi/12,-np.pi/6),10,f0,2)
floor = W.CheckeredSphereObject(np.array((0,-90001-600,0)),f0,np.array((1,0,0)),90000)

sphere = W.SphereObject(np.array((200,.6-100,1)),f1,np.array((0,1,1)),.6)
sphere2 = W.SphereObject(np.array((1,.8-100,1)),f0,np.array((1,1,0)),.6)
s3 = W.SphereObject(np.array((2,5-100,1)),f0,np.array((0,0,1)),3)

b1 = W.MeshObject(np.array((-5,4-100,1)),f0,mesh.Mesh.from_file('models/block100.stl'),np.array((0,1,0)))
b2 = W.MeshObject(np.array((100,4-100,100)),f3,mesh.Mesh.from_file('models/block100.stl'),np.array((.8,.5,0)))
scene = W.Scene(cam, np.array((30, 100, -30)), [b1,b2,floor])

scene.render().show()