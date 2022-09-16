from lib import World as W
import numpy as np
from stl import mesh

fp_out = "outputs/final/"

campos = (0, 50, -700)
camrot = (0, -np.pi / 7)
lpos = np.array((3000, 10000, -3000))

if __name__ == '__main__':
    v = 0
    numobjs = 11
    offset = 2400

    y = 1 / np.sqrt(1 - v ** 2)
    print(f"gamma: {y}")

    gap, step = np.linspace(-3000 + offset, 3000 + offset, numobjs, dtype=W.np_type, retstep=True)

    f0 = W.Frame([0, 0, 0])
    f1 = W.Frame([-v * W.c, 0, 0], f0)

    camera = W.Camera(0, campos, camrot, 10, f0, 2)

    objpos = (np.array([1, 0, 0], dtype=W.np_type)[:, np.newaxis] * gap * y).T
    objpos[:, 1] += -300
    print(objpos)
    objects = [W.MeshObject(objpos[i], f1, mesh.Mesh.from_file('models/cube_indent_100.stl'),
                            np.array((82.0 / 255.0, 110.0 / 255.0, 235.0 / 255.0))) for i in
               range(numobjs)]

    scene = W.Scene(camera, lpos, objects)

    scene.render().save(fp=fp_out + f"img{int(v * 10):02d}_reflecc.gif", format='PNG')
