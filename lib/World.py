import numpy as np

np_type = np.float64
c = 3e8

FARAWAY = 1.0e+39  # A large distance

def norm(arr): return arr/np.sqrt(np.sum(np.square(arr),axis=1))
def lt_velo(lt, velo):
    velo=velo[:,np.newaxis]
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
        lt_mat[0, 0] = g
        lt_mat[0, 1:] = lt_mat[1:, 0] = -b * g
        lt_mat[1:, 1:] += (g - 1) * np.matmul(b[np.newaxis].T, b[np.newaxis]) / b2

        assert(abs(np.linalg.det(lt_mat)-1)<1e-12)
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

        w, h = (640, 480)
        r = float(w) / h
        # Screen coordinates: x0, y0, x1, y1.
        S = (-1, 1 / r + .25, 1, -1 / r + .25)
        x = np.tile(np.linspace(S[0], S[2], w), h)
        y = np.repeat(np.linspace(S[1], S[3], h), w)

        # TODO: the time in this is almost definitely wrong, how do i specify a time such that after transform they
        #  are all the same time?
        self.screen_coords = np.stack((np.full((x.shape[0],), time*c), x, y, np.zeros(x.shape[0])), axis=0)
        self.ray_dirs = (self.screen_coords - self.point)[1:]

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


class Object:
    def __init__(self, position, frame):
        self.position = np.array(position, dtype=np_type)
        self.frame = frame

    def intersect_frame(self, screen):
        # TODO: add inverse transform from this back to screen frame (i brain die)
        lt = screen.frame.compute_lt_to_frame(self.frame)
        pt, dirs = screen.get_point_in_frame(self.frame), screen.get_ray_dirs_in_frame(self.frame)
        time, pos = pt[0], pt[1:]
        dists = self.intersect(pos, dirs)
        v4 = np.stack((time-(dists/c),pos+(dirs*dists)),axis=0)
        # add inv transform here before return pls
        return screen.get_point_from_frame(v4)

    def intersect(self, source, direction):
        return FARAWAY


class SphereObject(Object):
    def __init__(self, position, frame, radius):
        super().__init__(position, frame)
        self.radius = radius

    def intersect(self, source, direction): # this is refactored and likely broken btw just check
        b = 2 * np.dot(direction, source - self.position)
        c = np.abs(self.position) + np.abs(source) - 2 * np.dot(self.position, source) - (self.radius ** 2)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)


class MeshObject(Object):
    def __init__(self, position, frame, mesh):
        super().__init__(position, frame)
        self.mesh = mesh

    def intersect(self, source, direction):
        # TODO: @chas-card pls port over code
        pass

# testing

f0=Frame([.6*c,0,0])
f1=Frame([.8*c,0,0],f0)
f2=Frame([-.8*c,0,0],f1)
print(f0.from_world_lt)
print(f1.from_world_lt)
print(f2.from_world_lt)
print(lt_velo(f0.lt,np.array([0,.8*c,0])))