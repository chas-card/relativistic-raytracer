from lib import World as W
import numpy as np
from stl import mesh

from multiprocessing import Pool

fp_out = "outputs/train/"

camrot = (0, -np.pi / 10)
lpos = np.array((3000, 10000, -3000))
frame_dur = 5


def render_time(camt, campos, f, o):
    print(camt)
    camera_t = W.Camera(camt, campos, camrot, 10, f, 2)
    scene_t = W.Scene(camera_t, lpos, o)
    return scene_t.render()


if __name__ == '__main__':
    v = 0.3

    f1 = W.Frame([0, 0, 0])
    f2 = W.Frame([v * W.c, 0, 0])

    objects = [W.MeshObject([0, 0, 0], f1, mesh.Mesh.from_file('models/tunnel_window.stl'),
                            np.array((82.0 / 255.0, 110.0 / 255.0, 235.0 / 255.0))),
               W.MeshObject([-1500, 0, 0], f2, mesh.Mesh.from_file('models/train.stl'),
                            np.array((82.0 / 255.0, 212.0 / 255.0, 150.0 / 255.0))),
               W.MeshObject([100, -100, 350], f1, mesh.Mesh.from_file('models/ico50.stl'),
                            np.array((212.0 / 255.0, 114.0 / 255.0, 85.0 / 255.0)))]

    frames = [(f1, [0, 120, -500], "tunnel"), (f2, [-1500, 120, -500], "train")]

    for f0, campos, nm in frames:
        camera = W.Camera(0, campos, camrot, 10, f0, 2)

        y = 1 / np.sqrt(1 - v ** 2)
        print(f"gamma: {y}")

        step = 2400
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
        p = Pool(processes=10)
        imgs = p.starmap(render_time, args)

        # for i in tqdm(range(t.shape[0]-1)):
        #    camera.set_time(t[i])
        #    imgs.append(scene.render())

        imgs[0].save(fp=fp_out + f"img_{nm}.gif", format='GIF', append_images=imgs[1:], save_all=True, duration=frame_dur,
                     loop=0)
