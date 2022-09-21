import math
from array import array

import numpy as np
from PIL import Image

from multiprocessing import Pool

np_type = np.float64
c = 8.0  # 3e8

FARAWAY = 1.0e+39  # A large distance


def norm(arr): return arr / np.sqrt(np.sum(np.square(arr), axis=0))


def lt_velo(lt, velo):
    if len(np.shape(velo)) == 1:
        velo = velo[:, np.newaxis]
    v4 = np.concatenate((np.array([c * np.ones(np.shape(velo)[1])]), velo), axis=0)
    vp4 = lt @ v4
    return vp4[1:] * c / vp4[0]


# frame class to handle defining different reference frames with respect to other reference frames
# world frame is just (0, 0, 0, 0) position, use it as a "special" frame to transform between any frames
class Frame:
    def __init__(self, velocity, ref=None):
        self.velocity = np.array(velocity, dtype=np_type)
        self.ref = ref

    @property
    def lt(self):
        b = self.velocity / c
        b2 = np.sum(np.square(b))
        assert (b2 <= 1)

        g = 1 / (np.sqrt(1 - b2))

        lt_mat = np.eye(4, dtype=np_type)
        if b2 == 0:
            return lt_mat

        lt_mat[0, 0] = g
        lt_mat[0, 1:] = lt_mat[1:, 0] = -b * g
        lt_mat[1:, 1:] += (g - 1) * np.matmul(b[np.newaxis].T, b[np.newaxis]) / b2

        assert (abs(np.linalg.det(lt_mat) - 1) < 1e-3)
        return lt_mat

    @property
    def inv_lt(self):
        inv_lt_mat = self.lt
        inv_lt_mat[0, 1:] = -inv_lt_mat[0, 1:]
        inv_lt_mat[1:, 0] = -inv_lt_mat[1:, 0]
        return inv_lt_mat

    @property
    def to_world_lt(self):
        if self.ref is None:
            return self.inv_lt
        else:
            return np.matmul(self.ref.to_world_lt, self.inv_lt)

    @property
    def from_world_lt(self):
        if self.ref is None:
            return self.lt
        else:
            return np.matmul(self.lt, self.ref.from_world_lt)

    def compute_lt_to_frame(self, frame):
        if not frame:
            return self.to_world_lt
        return np.matmul(frame.from_world_lt, self.to_world_lt)

    def compute_lt_from_frame(self, frame):
        if not frame:
            return self.from_world_lt
        return np.matmul(self.from_world_lt, frame.to_world_lt)

    def __str__(self):
        return "Frame with velocity " + str(self.velocity) + " wrt " + str(self.ref)


# Screen class acts as the camera from which rays are projected
class Camera:
    def __init__(self, time, pos, rotn, dof, frame, bounces, res=(640,480)):  # rotn (x axis, y axis)
        self.point = None
        self.w, self.h = res
        self.ray_dirs = None
        self.screen_coords = None
        self.time = time
        self.pos = pos
        self.rotn = rotn
        self.dof = dof
        self.frame = frame
        self.bounces = bounces
        self.calc_rays()

    def calc_rays(self):
        x, y, z = self.pos
        self.point = np.array((self.time * c, x, y, z + self.dof), dtype=np_type)

        r = float(self.w) / self.h
        # Screen coordinates: x0, y0, x1, y1.
        S = 10 * np.array((-1, 1 / r, 1, -1 / r))
        x = np.tile(np.linspace(S[0], S[2], self.w, dtype=np_type), self.h)
        y = np.repeat(np.linspace(S[1], S[3], self.h, dtype=np_type), self.w)

        sigma, theta = self.rotn
        rotmat = np.array([[1, 0, 0, 0], [0, np.cos(sigma), 0, np.sin(sigma)], [0, 0, 1, 0],
                           [0, -np.sin(sigma), 0, np.cos(sigma)]], dtype=np_type) @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta), np.sin(theta)], [0, 0, -np.sin(theta), np.cos(theta)]],
            dtype=np_type)
        self.screen_coords = rotmat @ np.stack((np.full((x.shape[0],), self.time * c), x, y, np.zeros(x.shape[0])),
                                               axis=0) + self.point[:, np.newaxis]
        self.point += rotmat @ np.array([0, 0, 0, -self.dof], dtype=np_type)

        self.ray_dirs = norm((self.screen_coords - self.point[:, np.newaxis])[1:])

    def set_time(self, t):
        self.time = t
        self.calc_rays()


class Scene:
    def __init__(self, camera, light, objs):
        self.camera = camera
        self.light = light
        self.objs = objs

    def raytrace(self, source, dirs, frame, bounce=0):
        res = [s.intersect_frame(source, dirs, frame) for s in self.objs]
        color = np.zeros((np.shape(source)[1], 3))
        if not res:
            return color
        dists, norms = zip(*res)  # (objects) x (screen dims)
        nearest = np.amin(dists, axis=0)
        for (s, d, n) in zip(self.objs, dists, norms):
            # print("Bounce "+str(bounce)+" of "+str(self.camera.bounces)+": Raytracing object "+str(s))
            hit = (nearest < 1e30) & (d == nearest)
            if np.any(hit):
                sourcec = np.compress(hit, source, axis=1)
                distsc = np.compress(hit, d)
                dirsc = np.compress(hit, dirs, axis=1)
                normsc = np.compress(hit, n, axis=1)
                colorc = s.light_frame(sourcec, dirsc, distsc, normsc, frame, self, bounce)
                ret = np.zeros((*hit.shape, 3))
                for i in range(3):
                    np.place(ret[:, i], hit, colorc[:, i])
                color += ret
        return color

    def trace_scene(self):
        return self.raytrace(np.array([self.camera.point] * np.shape(self.camera.ray_dirs)[1]).T, self.camera.ray_dirs,
                             self.camera.frame, bounce=0)

    def render(self):
        return Image.merge("RGB", [
            Image.fromarray((255 * np.clip(see, 0, 1).reshape((self.camera.h, self.camera.w))).astype(np.uint8), "L")
            for see in self.trace_scene().T]
                           )


class Object:

    def __init__(self, position, frame, diffuse, mirror=0.8):
        """

        :param array position: array of x y z position
        :param frame: the reference frame to define the position and velocity of the object
        :param array diffuse: RGB colour (from 0 to 1)
        :param float mirror: how much to reflect
        """
        self.position = np.array(position, dtype=np_type)
        self.frame = frame
        self.diffuse = np.array(diffuse, dtype=np_type)
        self.mirror = mirror

    def intersect_frame(self, source, dirs, frame):
        """

        :param frame: frame from which the source rays are projected from
        :param dirs: direction of the N rays cast from the N sources | shape(3, N)
        :param source: position and time of the N sources | shape(4, N)
        :return:
        """
        lt = frame.compute_lt_to_frame(self.frame)
        pt, dirs = lt @ source, -lt_velo(lt, -dirs * c) / c

        time, pos = pt[0], pt[1:]
        (dists, norms) = self.intersect(pos, dirs)

        v4 = np.concatenate(([-dists], (dirs * dists)), axis=0)
        return np.sqrt(np.sum(np.square(((frame.compute_lt_from_frame(self.frame) @ v4)[1:]).T), axis=1)), norms

    def light_frame(self, source, dirs, dists, norms, frame, scene, bounce):
        """

        :param int bounce: number of bounces remaining
        :param scene: scene
        :param frame: frame from which the source rays are projected from
        :param norms: object-frame face normals | shape(N, 3)
        :param dists: ray intersect distances | shape(N,)
        :param dirs: direction of the N rays cast from the N sources | shape(3, N)
        :param source: position and time of the N sources | shape(4, N)
        :return:
        """
        lt = frame.compute_lt_to_frame(self.frame)
        v4 = np.concatenate(([-dists], (dirs * dists)), axis=0)

        pt, dirs = lt @ source, -lt_velo(lt, -dirs * c) / c
        dists = np.sqrt(np.sum(np.square((lt @ v4)[1:]).T, axis=1))

        return self.light(pt, dirs, dists, norms, scene, bounce)

    def diffuseColor(self, M):
        """

        :param np.ndarray M: intersection point(s) | shape(N, 3)
        :return: colour(s)
        """
        return self.diffuse

    def intersect(self, source, direction):
        """
        Ray to Object Intersect function

        :param np.ndarray source: ray source position vector | shape(3, N)
        :param np.ndarray direction: rays direction unit vector | shape(3, N)
        :return: intersection distance for each ray  shape(N,)
        """
        return np.full(direction.shape[1], FARAWAY)  # default return array of FARAWAY (no intersect)

    def light(self, source, dirs, dists, norms, scene, bounce):
        """
        Recursive raytrace function

        :param np.ndarray source: position and time of the N sources | shape(4, N)
        :param np.ndarray dirs: direction of the N rays cast from the N sources | shape(3, N)
        :param np.ndarray dists: ray intersect distances | shape(N,)
        :param np.ndarray norms: object-frame face normals | shape(N, 3)
        :param scene: array of Object instances
        :param int bounce: number of bounces remaining
        :return: array of colours for each pixel | shape(N,3)
        """
        time = source[0] - dists
        pts = source[1:] + dirs * dists.T
        tol = self.dirs_to_thing(scene.light, np.concatenate([[time], pts], axis=0), scene.camera.frame)
        toc = self.dirs_to_thing(scene.camera.point[1:], np.concatenate([[time], pts], axis=0), scene.camera.frame)
        nudged = pts + norms * .0001  # default return all black

        # return np.array([self.diffuseColor(pts)]*len(dists))

        n4d = np.concatenate(([time], nudged), axis=0)  # TODO
        distsl = [s.intersect_frame(n4d, tol, self.frame)[0] for s in scene.objs]

        nearl = np.amin(distsl, axis=0)
        seelight = nearl > 1e30
        color = np.array([[.05] * 3] * len(dists))

        lv = np.maximum(np.einsum("ij,ij->j", norms, tol), 0.1)
        color += np.outer((lv * seelight), self.diffuseColor(pts))
        color += np.outer(lv, self.diffuseColor(pts))

        if bounce < scene.camera.bounces:
            nray = norm(dirs - 2 * norms * np.einsum("ij,ij->j", dirs, norms))
            color += scene.raytrace(n4d, nray, self.frame, bounce + 1) * self.mirror

        phong = np.einsum("ij,ij->j", norms, norm(tol + toc))
        # color += np.outer((np.power(np.clip(phong, 0, 1), 50)), np.ones(3))
        color += np.outer((np.power(np.clip(phong, 0, 1), 50) * seelight), np.ones(3))

        return color
        # return np.full(direction.shape[1], self.diffuseColor(None))    # default return all black

    def dirs_to_thing(self, thing, pos, thingframe):
        dirs = thing[:, np.newaxis] - (self.frame.compute_lt_to_frame(thingframe) @ pos)[1:]
        return -norm(lt_velo(self.frame.compute_lt_from_frame(thingframe), -dirs))

    def __str__(self):
        return self.__class__.__name__ + " at position " + str(self.position) + " with color " + str(
            self.diffuse) + " in frame: " + str(self.frame)


class SphereObject(Object):
    def __init__(self, position, frame, diffuse, radius):
        super().__init__(position, frame, diffuse)
        self.radius = radius

    def intersect(self, source, direction):  # this is refactored and likely broken btw just check
        b = 2 * np.einsum("ij,ij->j", direction, source - self.position[:, np.newaxis])
        see = np.sum(np.square(self.position)) + np.sum(np.square(source), axis=0) - 2 * np.dot(self.position,
                                                                                                source) - (
                      self.radius ** 2)

        disc = np.square(b) - (4 * see)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return (
            np.where(pred, h, FARAWAY),
            np.where(pred[np.newaxis, :],
                     (source - self.position[:, np.newaxis] + np.einsum("ij,j->ij", direction, h)) / self.radius,
                     np.zeros(np.shape(direction)))
        )


class CheckeredSphereObject(SphereObject):
    def diffuseColor(self, M):
        print(M)
        checker = ((M[0] * 2).astype(int) % 2) == ((M[2] * 2).astype(int) % 2)
        return self.diffuse * checker


def process_chunk(source, direction, meshN_chunk, v0_chunk, v1_chunk, v2_chunk):
    intersectLens = np.einsum("at,tb->ab", meshN_chunk, direction)

    intersectLens = np.where(intersectLens == 0, 1 / FARAWAY, intersectLens)

    t = np.einsum("ab,ab->ab",
                  np.einsum("abt,at->ab", (v0_chunk[:, np.newaxis, :] - source.T[np.newaxis, :, :]),
                            meshN_chunk), np.reciprocal(intersectLens))
    t = np.where(t < 0, FARAWAY, t)

    P = source.T[np.newaxis] + np.einsum("ab,cb->cba", direction, t)

    edge = (v1_chunk - v0_chunk)
    vp = P - v0_chunk[:, np.newaxis, :]
    if MeshObject.path1 is None:
        path_info = np.einsum_path("ijk,uj,uvk->uvi", MeshObject.eijk, edge, vp, optimize='optimal')
        MeshObject.path1 = path_info[0]
        # print(path_info[1])
    C = np.einsum("ijk,uj,uvk->uvi", MeshObject.eijk, edge, vp, optimize=MeshObject.path1)
    if MeshObject.path2 is None:
        path_info = np.einsum_path("ab,acb->ac", meshN_chunk, C, optimize='greedy')
        MeshObject.path2 = path_info[0]
        # print(path_info[1])
    d = np.einsum("ab,acb->ac", meshN_chunk, C, optimize=MeshObject.path2)
    t = np.where(d < 0, FARAWAY, t)

    edge = (v2_chunk - v1_chunk)
    vp = P - v1_chunk[:, np.newaxis, :]
    C = np.einsum("ijk,uj,uvk->uvi", MeshObject.eijk, edge, vp, optimize=MeshObject.path1)
    d = np.einsum("ab,acb->ac", meshN_chunk, C, optimize=MeshObject.path2)
    t = np.where(d < 0, FARAWAY, t)

    edge = (v0_chunk - v2_chunk)
    vp = P - v2_chunk[:, np.newaxis, :]
    C = np.einsum("ijk,uj,uvk->uvi", MeshObject.eijk, edge, vp, optimize=MeshObject.path1)
    d = np.einsum("ab,acb->ac", meshN_chunk, C, optimize=MeshObject.path2)
    t = np.where(d < 0, FARAWAY, t)

    return t


class MeshObject(Object):
    path1 = None
    path2 = None
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    def __init__(self, position, frame, mesh, diffuse, mirror=0.5, threads=0, chunk_size = 80):
        super().__init__(position, frame, diffuse, mirror=mirror)
        self.threads = threads
        self.m = mesh
        self.m.translate(position)
        self.chunk_size = chunk_size

    def intersect(self, source, direction):
        m = self.m
        meshN = m.get_unit_normals()

        # array of intersect lengths for ALL triangles
        t_overall = np.full(direction.shape[1], FARAWAY)  # initialize to assume all distances are FARAWAY
        N_overall = np.full(direction.shape, 0, dtype=np_type).T

        polygons = meshN.shape[0]
        N = math.ceil(polygons / self.chunk_size)

        chunks = [min((i + 1) * self.chunk_size, polygons) for i in range(N)]
        meshN_chunks = np.array_split(meshN, chunks, axis=0)
        v0_chunks = np.array_split(m.v0, chunks, axis=0)
        v1_chunks = np.array_split(m.v1, chunks, axis=0)
        v2_chunks = np.array_split(m.v2, chunks, axis=0)

        ts = []
        if self.threads == 0:
            for i in range(N):
                ts.append(
                    process_chunk(source, direction, meshN_chunks[i], v0_chunks[i], v1_chunks[i], v2_chunks[i]))
        else:
            p = Pool(processes=self.threads)
            ts = p.starmap(process_chunk,
                           [(source, direction, meshN_chunks[i], v0_chunks[i], v1_chunks[i], v2_chunks[i])
                            for i in range(N)]
                           )
            p.close()
        for i, t in enumerate(ts):
            min_t = np.min(t, axis=0)
            min_polygon = (t != FARAWAY) & (t == min_t[np.newaxis, :])
            b = min_polygon[:, :, np.newaxis]
            sel_n = np.sum(b * meshN_chunks[i][:, np.newaxis, :], axis=0)
            N_overall += sel_n
            t_overall = np.where(min_t < t_overall, min_t, t_overall)

        return t_overall, N_overall.T

    def chas_intersect(self, source, direction):
        # TODO: TEST IF WORKS
        # numpy-stl mesh get normal vectors as unit vectors
        m = self.m
        meshN = self.m.get_unit_normals()

        # array of intersect lengths for ALL triangles
        # initialize to assume all distances are FARAWAY
        t_overall = np.full(direction.shape[1], FARAWAY)

        # array of intersect normals for ALL triangles
        N_overall = np.full(direction.shape, 0).T

        direction = direction.T
        for i in range(0, len(m.v0)):
            v1 = m.v0[i]  # point 1
            v2 = m.v1[i]  # point 2
            v3 = m.v2[i]  # point 3
            N = meshN[i]

            # INTERSECT TRIANGLE PLANE =================================
            # compute intersect lengths to plane
            intersectLens = direction.dot(N)
            # Check if ray and plane are parallel
            if intersectLens.all() == 0:
                continue  # no intersects
            # intersect lengths to plane
            t = (v1 - source.T).dot(N) / intersectLens
            # check if triangle behind ray
            t = np.where(t < 0, FARAWAY, t)
            # intersection point(s) (individual vectors) using equation
            P = source.T + direction * t[:, np.newaxis]
            # END INTERSECT TRIANGLE PLANE =============================

            # CHECK INSIDE/OUTSIDE =====================================
            # whether intersect point within triangle area

            edge = v2 - v1  # vector 1-2
            vp = P - v1  # vector 1-P    array of such vectors
            C = np.cross(edge, vp)  # C IS N x 3
            t = np.where(np.dot(C, N) < 0, FARAWAY, t)

            edge = v3 - v2  # vector 2-3
            vp = P - v2  # vector 2-P    array of such vectors
            C = np.cross(edge, vp)  # C IS N x 3
            t = np.where(np.dot(C, N) < 0, FARAWAY, t)

            edge = v1 - v3  # vector 3-1
            vp = P - v3  # vector 3-P    array of such vectors
            C = np.cross(edge, vp)  # C IS N x 3
            t = np.where(np.dot(C, N) < 0, FARAWAY, t)

            # END CHECK INSIDE/OUTSIDE =================================

            # Get closest distances
            t_overall = np.where(t < t_overall, t, t_overall)
            # array of whether t < t_overall
            tLess_bool = t < t_overall

            # Get closest distances
            t_overall = np.where(tLess_bool, t, t_overall)

            # Add normals to N_overall (only if t was closer)
            tLess_bool_broadcast = np.repeat(tLess_bool.reshape((1, len(tLess_bool))), 3, axis=0).T
            N_overall = np.where(tLess_bool_broadcast, N, N_overall)

        return t_overall, N_overall
