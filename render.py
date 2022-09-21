from lib import World as W
import numpy as np
from stl import mesh

from multiprocessing import Pool

# testing
if __name__ == '__main__':
    f0=W.Frame([0,0,0])
    f1=W.Frame([-0.02*W.c,0,0],f0)
    f3=W.Frame([-0.007*W.c,0,0],f0)

    cam1 = W.Camera(0,(-80,50,-100),(np.pi/12,-np.pi/6),10,f0,3, res=(640,240))

    b1 = W.MeshObject(np.array((-5,5-70,120)), f1, mesh.Mesh.from_file('models/ico50.stl'), np.array((0, 1, 0)), mirror=0.1)
    #b2 = W.MeshObject(np.array((-5-80,5-30,60)), f3, mesh.Mesh.from_file('models/ico50.stl'), np.array((.8, .5, 0)), mirror=0.1)
    b3 = W.MeshObject(np.array((-5-120,5-70,220)), f3, mesh.Mesh.from_file('models/ico50.stl'), np.array((.4, 0, 0.8)), mirror=0.1)
    scene1 = W.Scene(cam1, np.array((1000, 2000, -1000)), [b1, b3])

    scene1.render().show()