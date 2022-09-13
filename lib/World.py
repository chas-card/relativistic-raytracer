from PIL import Image
import numpy as np

np_type = np.float32
c = 3e8

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
        self.point = np.array((time*c, 0, 0.35, -1), dtype=np_type)
        self.frame = frame

        self.w, self.h = w, h = (640, 480)
        r = float(w) / h
        # Screen coordinates: x0, y0, x1, y1.
        S = (-1, 1 / r + .25, 1, -1 / r + .25)
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
        return lt_velo(lt, self.ray_dirs)


class Scene:
    def __init__(self, camera, light, objs):
        self.camera = camera
        self.light = light
        self.objs = objs
    
    def raytrace(self, bounce=0):
        distances = [s.intersect_frame(self.camera) for s in self.objs] # (objects) x (screen dims)
        print(len(distances),distances[0].shape)
        nearest = np.amin(distances, axis=0)
        print(nearest.shape)
        color = np.array(np.zeros((*nearest.shape,3)))
        for (s, d) in zip(self.objs, distances):
            # print("pew")
            hit = (nearest != FARAWAY) & (d == nearest)
            if np.any(hit):
                distsc = np.extract(hit, d)
                dirsc = np.extract(hit, distances)
                cc = s.light_frame(self.camera, dirsc, distsc, self.objs, .5)
                ret = np.zeros((*hit.shape,3))
                print(np.shape(ret),np.shape(cc))
                for i in range(3):
                    print(sum(cc))
                    np.place(ret[:,i],hit,cc[:,i])
                color+=ret
        return color

class Object:

    def __init__(self, position, frame, diffuse, mirror=0.5):
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

    def intersect_frame(self, screen):
        """

        :param screen:
        :return:
        """
        lt = screen.frame.compute_lt_to_frame(self.frame)
        pt, dirs = screen.get_point_in_frame(self.frame), screen.get_ray_dirs_in_frame(self.frame)

        time, pos = pt[0], pt[1:]
        dists = self.intersect(pos, dirs)
        
        v4 = np.concatenate((np.array([time-(dists/c)]),pos[:,np.newaxis]+(dirs*dists)),axis=0)
        return np.sqrt(np.sum(np.square((screen.frame.compute_lt_from_frame(self.frame) @ v4)[1:].T), axis=1))

    def light_frame(self, screen, dirs, dists, scene, bounce):
        """

        :param screen:
        :return:
        """
        print("ping!")
        lt = screen.frame.compute_lt_to_frame(self.frame)
        pt, dirs = screen.get_point_in_frame(self.frame), screen.get_ray_dirs_in_frame(self.frame)
        time, pos = pt[0], pt[1:]
        print(self.light(pos, dirs, dists, scene, bounce))
        return self.light(pos, dirs, dists, scene, bounce)

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

        :param np.ndarray source: ray source position vector | shape(3,)
        :param np.ndarray direction: rays direction unit vector | shape(N,3)
        :return: intersection distance for each ray  shape(N,)
        """
        return np.full(direction.shape[1], FARAWAY)		# default return array of FARAWAY (no intersect)

    def light(self, source, direction, d, scene, bounce):
        """
        Recursive raytrace function

        :param np.ndarray source: ray source position vector | shape(3,)
        :param np.ndarray direction: rays direction unit vector | shape(N,3)
        :param np.ndarray d: ray intersect distances | shape(N,)
        :param scene: array of Object instances
        :param int bounce: number of bounces
        :return: array of colours for each pixel | shape(N,3)
        """
        return np.full(direction.shape[1], np.array([1,1,1]))    # default return all black


class SphereObject(Object):
    def __init__(self, position, frame, diffuse, radius):
        super().__init__(position, frame, diffuse)
        self.radius = radius

    def intersect(self, source, direction): # this is refactored and likely broken btw just check
        print(direction[:,0])
        print(source, self.position)
        b = 2 * np.dot(direction.T, source - self.position)
        print(np.shape(b))
        c = np.sum(np.square(self.position),axis=0) + np.sum(np.square(source),axis=0) - 2 * np.dot(self.position, source) - (self.radius ** 2)
        print(c)
        print(np.shape(c))
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        print(b[0],c,h0[0],h1[0])
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        print(np.sum(pred))
        return np.where(pred, h, FARAWAY)

    def light(self, source, direction, d, scene, bounce):
        return np.full((direction.shape[1],3), np.array([1,1,1]))    # default return all black


class MeshObject(Object):
    def __init__(self, position, frame, mesh, diffuse, mirror=0.5):
        super().__init__(position, frame, diffuse, mirror=mirror)
        self.m = mesh
        self.chunksize = 60

    def np_intersect(self, source, direction):
        m = self.m
        meshN = m.get_unit_normals()

        # array of intersect lengths for ALL triangles
        t_overall = np.full(direction.shape[1], FARAWAY)  # initialize to assume all distances are FARAWAY

        polygons = meshN.shape[0]
        N = polygons//self.chunksize

        chunks = [min((i+1)*self.chunksize, polygons) for i in range(N)]
        meshN_chunks = np.array_split(meshN, chunks, axis=0)
        v0_chunks = np.array_split(m.v0, chunks, axis=0)
        v1_chunks = np.array_split(m.v1, chunks, axis=0)
        v2_chunks = np.array_split(m.v2, chunks, axis=0)

        done_size = 0
        for i in range(N):
            curr_size = meshN_chunks[i].shape[0]
            intersectLens = np.einsum("at,tb->ab", meshN_chunks[i], direction)

            t = np.einsum("a,ab->ab", np.einsum("at,at->a", (v0_chunks[i] - source), meshN_chunks[i]), np.reciprocal(intersectLens))
            t = np.where(t < 0, FARAWAY, t)

            P = source + np.einsum("ab,cb->cba", direction, t)

            edge = (v1_chunks[i] - v0_chunks[i])[:, np.newaxis, :]
            vp = P - v0_chunks[i][:, np.newaxis, :]
            C = np.cross(edge, vp)
            d = np.einsum("ab,acb->ac", meshN_chunks[i], C)
            t = np.where(d < 0, FARAWAY, t)

            edge = (v2_chunks[i] - v1_chunks[i])[:, np.newaxis, :]
            vp = P - v1_chunks[i][:, np.newaxis, :]
            C = np.cross(edge, vp)
            d = np.einsum("ab,acb->ac", meshN_chunks[i], C)
            t = np.where(d < 0, FARAWAY, t)

            edge = (v0_chunks[i] - v2_chunks[i])[:, np.newaxis, :]
            vp = P - v2_chunks[i][:, np.newaxis, :]
            C = np.cross(edge, vp)
            d = np.einsum("ab,acb->ac", meshN_chunks[i], C)
            t = np.where(d < 0, FARAWAY, t)

            min_t = np.min(t, axis=0)
            done_size += curr_size
            t_overall = np.where(min_t < t_overall, min_t, t_overall)
            print(done_size)

        return t_overall

    def intersect(self, source, direction):
        # TODO: TEST IF WORKS
        # numpy-stl mesh get normal vectors as unit vectors
        m = self.m
        meshN = self.m.get_unit_normals()

        # array of intersect lengths for ALL triangles
        t_overall = np.full(direction.shape[1], FARAWAY)  # initialize to assume all distances are FARAWAY

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
                break			# no intersects
            # intersect lengths to plane
            t = (v1 - source).dot(N) / intersectLens
            # check if triangle behind ray
            t = np.where(t < 0, FARAWAY, t)
            # intersection point(s) (individual vectors) using equation
            P = source + direction * t[:,np.newaxis]
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

        return t_overall

    def light(self, source, direction, d, scene, bounce):
        return np.full(direction.shape[1], np.array([0,0,0]))    # default return all black



# testing

#f0=Frame([0,0,0])
f1=Frame([.6*c,0,0])
f2=Frame([.8*c,0,0],f1)
f3=Frame([-.8*c,0,0],f2)
print(f1.from_world_lt)
print(f2.from_world_lt)
print(f3.from_world_lt)
print(lt_velo(f1.lt,np.array([0,.8*c,0])))

cam = Screen(np.array((0, 0.35, -1)),0,f1)
sphere = SphereObject(np.array((.75,.1,1)),f1,np.array((0,0,1)),.6)
scene = Scene(cam, np.array((300, 1000, -300)), [sphere])
color = scene.raytrace()
rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((cam.h, cam.w))).astype(np.uint8), "L") for c in color.T]

Image.merge("RGB", rgb).show()