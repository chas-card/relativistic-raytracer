from lib import World as W
import numpy as np
from stl import mesh

from tqdm import tqdm

fp_out = "outputs/more095.gif"

v = 0.95
numobjs = 14
offset = 3400

y = 1/np.sqrt(1-v**2)
print(y)

gap, step = np.linspace(-3000 + offset, 3000 + offset, numobjs, dtype=W.np_type, retstep=True)
t = np.linspace(0, step/(v*W.c), 30)
end = t[-2]
dur = round(end/2/29)*29
print(dur)

f0 = W.Frame([0, 0, 0])
f1 = W.Frame([-v * W.c, 0, 0], f0)

camera = W.Camera(0, (0, 50, -700), (2*np.pi / 7, -np.pi / 7), 10, f0, 2)

objpos = (np.array([1, 0, 0], dtype=W.np_type)[:, np.newaxis] * gap*y).T
objpos[:, 1] += -300
print(objpos)
objects = [W.MeshObject(objpos[i], f1, mesh.Mesh.from_file('models/cube_indent_100.stl'), np.array((0, 1, 0))) for i in
           range(numobjs)]

scene = W.Scene(camera, np.array((3000, 10000, -3000)), objects)

scene.render().show()

imgs = []
for i in tqdm(range(t.shape[0]-1)):
    camera.set_time(t[i])
    imgs.append(scene.render())

imgs[0].save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=dur, loop=0)
