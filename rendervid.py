from lib import World as W
import numpy as np
from stl import mesh

from multiprocessing import Pool

fp_out = "outputs/final/"

campos = (0, 50, -900)
camrot = (0, -np.pi / 7)
lpos = np.array((3000, 10000, -3000))
frame_dur = 5


def render_time(camt, f, o):
    print(camt)
    camera_t = W.Camera(camt, campos, camrot, 10, f, 2, res=(640,240))
    scene_t = W.Scene(camera_t, lpos, o)
    return scene_t.render()


if __name__ == '__main__':
    vrange = np.linspace(0.1, 0.9, 5)
    print(vrange)

    f0 = W.Frame([0, 0, 0])

    camera = W.Camera(0, campos, camrot, 10, f0, 2, res=(640,240))

    for v in vrange:
        numobjs = 8
        offset = 1000

        y = 1 / np.sqrt(1 - v ** 2)
        print(f"gamma: {y}")

        gap, step = np.linspace(-3000 + offset, 3000 + offset, numobjs, dtype=W.np_type, retstep=True)
        tstep = step / (v * W.c)
        print(tstep)
        num_frames = round(tstep/frame_dur)
        t = np.linspace(0, step / (v * W.c), num_frames, dtype=W.np_type)
        print(t)
        t = t[:-1:]
        end = t[-1]
        print(f"interval time: {end}")
        print(f"nfames: {num_frames}")

        f1 = W.Frame([-v * W.c, 0, 0], f0)
        objpos = (np.array([1, 0, 0], dtype=W.np_type)[:, np.newaxis] * gap * y).T
        objpos[:, 1] += -300
        print(objpos)
        objects = [W.MeshObject(objpos[i], f1, mesh.Mesh.from_file('models/cube_indent_100.stl'),
                                np.array((82.0 / 255.0, 110.0 / 255.0, 235.0 / 255.0))) for i in
                   range(numobjs)]

        scene = W.Scene(camera, lpos, objects)

        args = [(t[i], f0, objects) for i in range(t.shape[0])]
        p = Pool(processes=10)
        imgs = p.starmap(render_time, args)

        imgs[0].save(fp=fp_out + f"imgg{int(v * 10):02d}.gif", format='GIF', append_images=imgs[1:], save_all=True, duration=frame_dur,
                     loop=0)
