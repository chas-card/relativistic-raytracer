import numpy as np

np_type = np.float64
c = 3 * 10e8

FARAWAY = 1.0e+39  # A large distance


# frame class to handle defining different reference frames with respect to other reference frames
# world frame is just (0, 0, 0, 0) position, use it as a "special" frame to transform between any frames
class Frame:
    def __init__(self, velocity, position, time, ref):
        self.velocity = np.array(velocity, dtype=np_type)
        self.position = np.array(position, dtype=np_type)
        self.time = time
        self.ref = ref

    @property
    def lt(self):
        b2 = np.sum(np.square(self.velocity))
        assert (b <= 1)

        g = 1 / (np.sqrt(1 - b2))

        lt_mat = np.eye(4, dtype=np_type)
        lt_mat[0, 0] = g
        lt_mat[0, 1:] = lt_mat[1:, 0] = -self.velocity * g / c
        lt_mat[1:, 1:] += (g - 1) * np.matmul(self.velocity[np.newaxis].T, self.velocity[np.newaxis]) / b2

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
            return np.eye(4, dtype=np_type)
        else:
            return np.matmul(self.inv_lt, self.ref.to_world)

    @property
    def from_world_lt(self):
        if self.ref is None:
            return np.eye(4, dtype=np_type)
        else:
            return np.matmul(self.ref.from_world, self.ref.lt)

    def compute_lt_to_frame(self, frame):
        return np.matmul(self.to_world_lt, frame.from_world_lt)

    def compute_lt_from_frame(self, frame):
        return np.matmul(frame.to_world_lt, self.from_world_lt)


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
        self.screen_coords = np.stack((np.full((x.shape[0],), time*c), x, y, np.zeros(x.shape[0])), axis=1)
        self.ray_dirs = self.screen_coords - self.point

    # get the "eye" point in any other frame
    def get_point_in_frame(self, toframe):
        lt = self.frame.compute_lt_to_frame(toframe)
        return np.matmul(lt, self.point)

    # get the coords of the screen in any other frame
    def get_screen_coords_in_frame(self, toframe):
        lt = self.frame.compute_lt_to_frame(toframe)
        return np.matmul(lt, self.screen_coords)

    # get the projected ray directions from any other frame (3D vector!)
    # TODO: check this (not sure whether it is ok to discard time info for the coord points)
    def get_ray_dirs_in_frame(self, toframe):
        coords = self.get_screen_coords_in_frame(toframe)
        pt = self.get_point_in_frame(toframe)
        dirs = coords - pt
        return dirs[1:]


class Object:
    def __init__(self, position, frame):
        self.position = np.array(position, dtype=np_type)
        self.frame = frame

    def intersect(self, source, direction):
        return FARAWAY


class SphereObject(Object):
    def __init__(self, position, frame, radius):
        super().__init__(position, frame)
        self.radius = radius

    def intersect_frame(self, source, direction, screen):
        # TODO: add inverse transform from this back to screen frame (i brain die)
        pt = screen.get_point_in_frame()
        time = pt[0]
        pos = pt[1:]

        # add inv transform here before return pls
        return self.intersect(pos, screen.get_ray_dirs_in_frame())

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

    def intersect_frame(self, source, direction, screen):
        # TODO: add inverse transform from this back to screen frame (i brain die)
        return self.intersect(screen.get_point_in_frame(), screen.get_ray_dirs_in_frame())

    def intersect(self, source, direction):
        # TODO: @chas-card pls port over code
        pass
