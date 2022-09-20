from lib import World as W
import numpy as np
from stl import mesh

from multiprocessing import Pool

fp_out = "outputs/train/"

camrot = (-np.pi/6, -np.pi / 10)
lpos = np.array((3000, 10000, -3000))
frame_dur = 5

offset = -2000
offset1 = 200
dist = 4200
doors_shown = True


def render_time(camt, campos, f, o):
    print(camt)
    camera_t = W.Camera(camt, campos, camrot, 10, f, 2)
    scene_t = W.Scene(camera_t, lpos, o)
    return scene_t.render()


if __name__ == '__main__':
    v = 0.7

    doorv = -7.2
    print(doorv)
    f1 = W.Frame([0, 0, 0])
    f2 = W.Frame([v * W.c, 0, 0])
    f3 = W.Frame([0, doorv, 0])

    objects = [W.MeshObject([0, 0, 0], f1, mesh.Mesh.from_file('models/tunnel_window.stl'),
                            np.array((82.0 / 255.0, 110.0 / 255.0, 235.0 / 255.0)), mirror=0.3),
               W.MeshObject([offset, 0, 0], f2, mesh.Mesh.from_file('models/train.stl'),
                            np.array((82.0 / 255.0, 212.0 / 255.0, 150.0 / 255.0)), mirror=0.3),
               W.MeshObject([100, -60, -250], f1, mesh.Mesh.from_file('models/ico50.stl'),
                            np.array((212.0 / 255.0, 114.0 / 255.0, 85.0 / 255.0)), mirror=0.3)]
    if (doors_shown):
        objects.append(W.MeshObject([0, dist, 0], f3, mesh.Mesh.from_file('models/doors.stl'),
                            np.array((199.0 / 255.0, 212.0 / 255.0, 85.0 / 255.0)), mirror=0.3))

    frames = [(f1, [0+offset1, 120, -500], "tunnel"), (f2, [offset+offset1, 120, -500], "train")]

    for f0, campos, nm in frames:
        camera = W.Camera(0, campos, camrot, 10, f0, 3)

        y = 1 / np.sqrt(1 - v ** 2)
        print(f"gamma: {y}")

        step = 3800
        tstep = step / (v * W.c)
        print(tstep)
        num_frames = round(tstep/frame_dur)
        t = np.linspace(0, step / (v * W.c), num_frames, dtype=W.np_type)
        print(t)
        t = t[:-1:]
        end = t[-1]
        print(f"interval time: {end}")
        print(f"nfames: {num_frames}")

        scene = W.Scene(camera, lpos, objects)

        #scene.render().show()

        args = [(t[i], campos, f0, objects) for i in range(t.shape[0])]

        p = Pool(processes=8)
        imgs = p.starmap(render_time, args)

        #imgs = []
        #for i in range(t.shape[0]):
        #    camt = t[i]
        #    f = f0
        #    o = objects
        #    print(camt)
        #    camera_t = W.Camera(camt, campos, camrot, 10, f, 2)
        #    scene_t = W.Scene(camera_t, lpos, o)
        #    imgs.append(scene_t.render())

        p.close()

        imgs[0].save(fp=fp_out + f"img{'door' if doors_shown else ''}_{nm}_fs_reflecc.gif", format='GIF', append_images=imgs[1:], save_all=True, duration=frame_dur,
                     loop=0)

