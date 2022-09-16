from lib import World as W
import numpy as np
from stl import mesh

# testing

f0=W.Frame([0,0,0])

cam = W.Camera(0,(-5,5,-6),(np.pi/12,-np.pi/8),3,f0,2)
floor = W.CheckeredSphereObject(np.array((0,-9000,0)),f0,np.array((1,0,0)),9000)

sph = [W.SphereObject(np.array((i,1,j)),f0,np.array((0,.3,.3)),1) for i in range(-5,6,2) for j in range(-3,11,2)]


scene = W.Scene(cam,np.array((50,50,50)),[*sph,floor])

scene.render().show()