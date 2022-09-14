from PIL import Image
import numpy as np
import math

np_type = np.float32
c = 8 #3e8

FARAWAY = 1.0e+39  # A large distance

def norm(arr): return arr/np.sqrt(np.sum(np.square(arr),axis=0))
def lt_velo(lt, velo):
    if len(np.shape(velo))==1: velo=velo[:,np.newaxis]
    v4 = np.concatenate((np.array([c*np.ones(np.shape(velo)[1])]), velo), axis=0)
    vp4 = lt @ v4
    return vp4[1:]*c/vp4[0]

# frame class to handle defining different reference frames with respect to other reference frames
# world frame is just (0, 0, 0, 0) position, use it as a "special" frame to transform between any frames
class Frame:
    def __init__(self, velocity, ref=None):
        self.velocity = np.array(velocity, dtype=np_type)
        self.ref = ref

    @property
    def lt(self):
        b=self.velocity/c
        b2 = np.sum(np.square(b))
        assert (b2 <= 1)

        g = 1 / (np.sqrt(1 - b2))

        lt_mat = np.eye(4, dtype=np_type)
        if b2==0: return lt_mat

        lt_mat[0, 0] = g
        lt_mat[0, 1:] = lt_mat[1:, 0] = -b * g
        lt_mat[1:, 1:] += (g - 1) * np.matmul(b[np.newaxis].T, b[np.newaxis]) / b2

        assert(abs(np.linalg.det(lt_mat)-1)<1e-3)
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
        return np.matmul(frame.from_world_lt, self.to_world_lt)

    def compute_lt_from_frame(self, frame):
        return np.matmul(self.from_world_lt, frame.to_world_lt)

# Screen class acts as the camera from which rays are projected
# TODO: make screen class work for arbitratily defined screen coords
class Screen:
    def __init__(self, res, time, frame):
        self.point = np.array((time*c, 0, 1, -10), dtype=np_type)
        self.frame = frame

        self.w, self.h = w, h = (640, 480)
        r = float(w) / h
        # Screen coordinates: x0, y0, x1, y1.
        S = 10 * np.array((-1, 1 / r + .25, 1, -1 / r + .25))
        x = np.tile(np.linspace(S[0], S[2], w), h)
        y = np.repeat(np.linspace(S[1], S[3], h), w)

        # TODO: the time in this is almost definitely wrong, how do i specify a time such that after transform they
        #  are all the same time?
        self.screen_coords = np.stack((np.full((x.shape[0],), time*c), x, y, np.zeros(x.shape[0])), axis=0)
        self.ray_dirs = norm((self.screen_coords - self.point[:,np.newaxis])[1:])

    # get the "eye" point in any other frame
    def get_point_in_frame(self, toframe):
        lt = self.frame.compute_lt_to_frame(toframe)
        return np.matmul(lt, self.point)

    def get_point_from_frame(self, fromframe):
        lt = self.frame.compute_lt_from_frame(fromframe)
        return np.matmul(lt, self.point)

    # get the coords of the screen in any other frame
    def get_screen_coords_in_frame(self, toframe):
        lt = self.frame.compute_lt_to_frame(toframe)
        return np.matmul(lt, self.screen_coords)

    # get the projected ray directions from any other frame (3D vector!)
    # TODO: check this (not sure whether it is ok to discard time info for the coord points)
    def get_ray_dirs_in_frame(self, toframe):
        lt = self.frame.compute_lt_to_frame(toframe)
        return lt_velo(lt, self.ray_dirs*c)/c


class Scene:
    def __init__(self, camera, light, objs):
        self.camera = camera
        self.light = light
        self.objs = objs
    
    def raytrace(self, source, dirs, frame, bounce=0):
        dists, norms = zip(*[s.intersect_frame(source, dirs, frame) for s in self.objs]) # (objects) x (screen dims)
        nearest = np.amin(dists, axis=0)
        color = np.array(np.zeros((*nearest.shape,3)))
        for (s, d, n) in zip(self.objs, dists, norms):
            hit = (nearest < 1e30) & (d == nearest)
            if np.any(hit):
                sourcec = np.compress(hit, source, axis=1)
                distsc = np.compress(hit, d)
                dirsc = np.compress(hit, dirs,axis=1)
                normsc = np.compress(hit, n, axis=1)
                print(np.shape(dirsc),np.shape(distsc),np.shape(normsc),"aaa")
                colorc = s.light_frame(sourcec, dirsc, distsc, normsc, frame, self, bounce)
                ret = np.zeros((*hit.shape,3))
                for i in range(3):
                    np.place(ret[:,i],hit,colorc[:,i])
                color+=ret
        return color

    def tracescene(self):
        return self.raytrace(np.array([self.camera.point]*np.shape(self.camera.ray_dirs)[1]).T,self.camera.ray_dirs,self.camera.frame,bounce=0)
class Object:

    def __init__(self, position, frame, diffuse, mirror=0.8):
        """
        Initialize new Object

        :param array position: array of x y z position
        :param Frame frame:
        :param array diffuse: RGB colour (from 0 to 1)
        :param float mirror: how much to reflect
        """
        self.position = np.array(position, dtype=np_type)
        self.frame = frame
        self.diffuse = np.array(diffuse, dtype=np_type)
        self.mirror = mirror

    def intersect_frame(self, source, dirs, frame):
        """

        :param screen:
        :return:
        """
        lt = frame.compute_lt_to_frame(self.frame)
        print(source)
        pt, dirs = lt @ source, lt_velo(lt, dirs*c)/c

        time, pos = pt[0], pt[1:]
        (dists, norms) = self.intersect(pos, dirs)
        
        v4 = np.concatenate((np.array([time-(dists/c)]),pos+(dirs*dists)),axis=0)
        return (np.sqrt(np.sum(np.square((frame.compute_lt_from_frame(self.frame) @ v4)[1:].T), axis=1)), norms)

    def light_frame(self, source, dirs, dists, norms, frame, scene, bounce):
        """

        :param screen:
        :return:
        """
        print("ping!")
        lt = frame.compute_lt_to_frame(self.frame)

        pt, dirs = lt @ source, lt_velo(lt, dirs*c)/c
        time, pos = pt[0], pt[1:]
        
        print(np.shape(dirs),np.shape(dists),np.shape(pos))
        v4 = np.concatenate(([time],np.einsum("ij,j->ij",dirs,dists)-pos),axis=0)
        print(np.shape(v4))
        dists = np.sqrt(np.sum(np.square((lt @ v4)[1:].T), axis=1))
        
        return self.light(pos, dirs, dists, norms, scene, bounce)

    def diffuseColor(self, M):
        """
        Object colour function

        :param np.ndarray M: intersection point(s)  shape(N,3)
        :return: colour(s)
        """
        return self.diffuse

    def intersect(self, source, direction):
        """
        Ray to Object Intersect function

        :param np.ndarray source: ray source position vector    | shape(3,N)
        :param np.ndarray direction: rays direction unit vector | shape(N,3)
        :return: tuple with (
            intersection distance for each ray          | shape(N,)
            intersection normals for each ray           | shape(N,3)
            )
        """
        return np.full(direction.shape[1], FARAWAY)		# default return array of FARAWAY (no intersect)

    def light(self, source, dirs, dists, norms, scene, bounce):
        """
        Recursive raytrace function

        :param np.ndarray source: ray source position vector | shape(N,3)
        :param np.ndarray dirs: rays direction unit vector | shape(N,3)
        :param np.ndarray dists: ray intersect distances | shape(N,)
        :param np.ndarray norms: object-frame face normals | shape(N,)
        :param scene: array of Object instances
        :param int bounce: number of bounces
        :return: array of colours for each pixel | shape(N,3)
        """
        print(np.shape(dirs*dists))
        pts = source + dirs*(dists.T)
        tol = norm(scene.light[:,np.newaxis] - pts)
        toc = norm(source - pts)
        nudged = pts + norms*.0001

        n4d = np.concatenate((np.array([np.zeros(np.shape(nudged)[1])]), nudged), axis=0) #TODO
        distsl = [s.intersect_frame(n4d,tol,self.frame)[0] for s in scene.objs]
        nearl = np.amin(distsl,axis=0)
        seelight = distsl[scene.objs.index(self)] == nearl

        color = np.array([[.05]*3]*len(dists))

        lv = np.maximum(np.einsum("ij,ij->j",norms,tol),0)
        color+=np.outer((lv * seelight),self.diffuseColor(pts))

        if bounce<1:
            nray = norm(dirs - 2 * norms * np.einsum("ij,ij->j",dirs,norms))
            color += scene.raytrace(n4d, nray, self.frame, bounce+1) * self.mirror

        phong = np.einsum("ij,ij->j",norms, norm(tol+toc))
        color += np.outer((np.power(np.clip(phong, 0, 1), 50) * seelight),np.ones(3))

        return color
        #return np.full(direction.shape[1], self.diffuseColor(None))    # default return all black


class SphereObject(Object):
    def __init__(self, position, frame, diffuse, radius):
        super().__init__(position, frame, diffuse)
        self.radius = radius

    def intersect(self, source, direction): # this is refactored and likely broken btw just check
        b = 2 * np.einsum("ij,ij->j", direction, source - self.position[:,np.newaxis])
        c = np.sum(np.square(self.position),axis=0) + np.sum(np.square(source),axis=0) - 2 * np.dot(self.position, source) - (self.radius ** 2)

        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        print(np.shape(source),np.shape(self.position),np.shape(h),np.shape(direction.T))
        return (
            np.where(pred, h, FARAWAY), 
            np.where(pred[np.newaxis,:], (source-self.position[:,np.newaxis]+np.einsum("ij,j->ij",direction,h)) / self.radius, np.zeros(np.shape(direction)))
        )

    #def light(self, source, dirs, dists, norms, scene, bounce):
        #return np.full((dirs.shape[1],3), self.diffuseColor(None))    # default return all black


class MeshObject(Object):
    def __init__(self, position, frame, mesh, diffuse, mirror=0.5):
        super().__init__(position, frame, diffuse, mirror=mirror)
        self.m = mesh
        self.m.translate(position)
        self.chunksize = 50

    def intersect(self, source, direction):
        m = self.m
        meshN = m.get_unit_normals()

        # array of intersect lengths for ALL triangles
        t_overall = np.full(direction.shape[1], FARAWAY)  # initialize to assume all distances are FARAWAY
        N_overall = np.full(direction.shape, 0, dtype=np_type).T

        polygons = meshN.shape[0]
        N = math.ceil(polygons / self.chunksize)

        chunks = [min((i + 1) * self.chunksize, polygons) for i in range(N)]
        meshN_chunks = np.array_split(meshN, chunks, axis=0)
        v0_chunks = np.array_split(m.v0, chunks, axis=0)
        v1_chunks = np.array_split(m.v1, chunks, axis=0)
        v2_chunks = np.array_split(m.v2, chunks, axis=0)

        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

        done_size = 0
        path = None
        for i in range(N):
            curr_size = meshN_chunks[i].shape[0]
            intersectLens = np.einsum("at,tb->ab", meshN_chunks[i], direction)

            intersectLens = np.where(intersectLens == 0, 1 / FARAWAY, intersectLens)

            t = np.einsum("ab,ab->ab",
                          np.einsum("abt,at->ab", (v0_chunks[i][:, np.newaxis, :] - source.T[np.newaxis, :, :]),
                                    meshN_chunks[i]), np.reciprocal(intersectLens))
            t = np.where(t < 0, FARAWAY, t)

            P = source.T[np.newaxis] + np.einsum("ab,cb->cba", direction, t)

            edge = (v1_chunks[i] - v0_chunks[i])
            vp = P - v0_chunks[i][:, np.newaxis, :]
            if path is None:
                path = np.einsum_path('ijk,uj,uvk->uvi', eijk, edge, vp, optimize='optimal')[0]
            C = np.einsum('ijk,uj,uvk->uvi', eijk, edge, vp, optimize=path)
            d = np.einsum("ab,acb->ac", meshN_chunks[i], C, optimize='greedy')
            t = np.where(d < 0, FARAWAY, t)

            edge = (v2_chunks[i] - v1_chunks[i])
            vp = P - v1_chunks[i][:, np.newaxis, :]
            C = np.einsum('ijk,uj,uvk->uvi', eijk, edge, vp, optimize=path)
            d = np.einsum("ab,acb->ac", meshN_chunks[i], C, optimize='greedy')
            t = np.where(d < 0, FARAWAY, t)

            edge = (v0_chunks[i] - v2_chunks[i])
            vp = P - v2_chunks[i][:, np.newaxis, :]
            C = np.einsum('ijk,uj,uvk->uvi', eijk, edge, vp, optimize=path)
            d = np.einsum("ab,acb->ac", meshN_chunks[i], C, optimize='greedy')
            t = np.where(d < 0, FARAWAY, t)

            min_t = np.min(t, axis=0)
            done_size += curr_size
            min_polygon = (t != FARAWAY) & (t == min_t[np.newaxis, :])
            b = min_polygon[:, :, np.newaxis]
            sel_n = np.sum(b * meshN_chunks[i][:, np.newaxis, :], axis=0)
            N_overall += sel_n
            t_overall = np.where(min_t < t_overall, min_t, t_overall)

        return t_overall, N_overall

    def old_intersect(self, source, direction):
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
                continue			# no intersects
            # intersect lengths to plane
            t = (v1 - source.T).dot(N) / intersectLens
            # check if triangle behind ray
            t = np.where(t < 0, FARAWAY, t)
            # intersection point(s) (individual vectors) using equation
            P = source.T + direction * t[:,np.newaxis]
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

            # array of whether t < t_overall
            tLess_bool = t < t_overall

            # Get closest distances
            t_overall = np.where(tLess_bool, t, t_overall)

            # Add normals to N_overall (only if t was closer)
            tLess_bool_broadcast = np.repeat(tLess_bool.reshape((1, len(tLess_bool))), 3, axis=0).T
            N_overall = np.where(tLess_bool_broadcast, N, N_overall)

        return (t_overall, N_overall)