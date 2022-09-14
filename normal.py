from lib import World as W
import numpy as np
from stl import mesh

# testing

f0=W.Frame([0,0,0])

cam = W.Camera(0,(0,5,-3),(0,-np.pi/12),3,f0,2)
floor = W.CheckeredSphereObject(np.array((0,-9000,0)),f0,np.array((1,0,0)),9000)

sph = [W.SphereObject(np.array((i,0,j)),f0,np.array((0,1,1)),1) for i in range(-5,6,2) for j in np.arange(0,11,2)]


scene = W.Scene(cam,np.array((5,5,-10)),[*sph,floor])

scene.render().show()