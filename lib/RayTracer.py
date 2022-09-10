""" ray tracer for spheres, triangles and meshes


"""

import math
import numpy as np
import matplotlib.pyplot as plt
import numbers
import time
from PIL import Image
from functools import reduce
from stl import mesh

# INDICATOR INTEGERS
# 0     Sphere
# 1     Triangle
# 2     Mesh

def extract(cond, x):
	if isinstance(x, numbers.Number):
		return x
	else:
		return np.extract(cond, x)

# Vec3

# *				only multiply with scalar
# +				adds vector
# -				subtract vector
# dot			dot product with other vector, returns scalar
# cross			cross product with other vector, returns vec3
# abs			SQUARED magnitude of vector
# norm			normalize vector to length 1
# components	tuple with x, y, z
# extract
class vec3():
	"""
	Array of 3D vector

	...
	Attributes
	----------
	x : np.array
	y : np.array
	"""
	def __init__(self, x, y, z):
		(self.x, self.y, self.z) = (x, y, z)

	def __mul__(self, other):
		return vec3(self.x * other, self.y * other, self.z * other)

	def __add__(self, other):
		return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

	def __sub__(self, other):
		return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

	def __len__(self):
		return len(self.x)

	def dot(self, other):
		return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

	def cross(self, other):
		v = np.cross([self.x, self.y, self.z], [other.x, other.y, other.z], axisa=0, axisb=0, axisc=0)
		return vec3(v[0], v[1], v[2])
	
	def ontoproj(self, other): #projects other vector onto self!
		return self * (self.dot(other)/(abs(other)**.5))

	def __abs__(self):
		return self.dot(self)

	def norm(self):
		mag = np.sqrt(abs(self))
		return self / np.where(mag == 0, 1, mag)

	def components(self):
		return (self.x, self.y, self.z)

	def extract(self, cond):
		return vec3(extract(cond, self.x),
					extract(cond, self.y),
					extract(cond, self.z))

	def place(self, cond):
		r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
		np.place(r.x, cond, self.x)
		np.place(r.y, cond, self.y)
		np.place(r.z, cond, self.z)
		return r

	def to4d(self, t):
		return np.array((*self.components(),t*np.ones(len(self))))

	def tobases(self): # returns matrix collection thingy; len x 3 x 3
		b = self.norm()
		i = vec3(np.ones(len(self)),np.zeros(len(self)),np.zeros(len(self)))
		j = vec3(np.zeros(len(self)),np.ones(len(self)),np.zeros(len(self)))
		ax0 = (self-self.ontoproj(np.where(i.dot(self)<j.dot(self), i, j))).norm()
		ax1 = b.cross(ax0) # mademadics
		return np.transpose(np.arr((b.components(),ax0.components(),ax1.components())),(2,0,1))
		# vec, xyz, loc

	def __str__(self):
		return "Vec3| x: " + str(self.x) + " y: " + str(self.y) + " z: " + str(self.z)

	def __repr__(self):
		return "Vec3| x: " + str(self.x) + " y: " + str(self.y) + " z: " + str(self.z)

class velo(vec3): 
	"""
	Array of velocities I guess?
	"""
	def __init__(self,x,y,z):
		super().__init__(x,y,z)
		b, bn = np.sqrt(abs(self)), self.norm()
		assert(all(abs(b)<1)) #what's the array equiv of this
		self.g = g = np.sqrt(1-b)
		c = np.pad(self.tobases(),((0,0),(0,1),(0,1)),'constant',0)
		c[:,-1,-1]=1
		lt = np.repeat([np.eye(4)],len(self))
		lt[:,0,0]=lt[:,-1,-1]=g
		lt[:,-1,0]=lt[:,0,-1]=-g*b
		self.lt = c @ lt @ np.inv(c) #4d ofc

	def __neg__(self):
		return velo(-self.x,-self.y,-self.z)

	#def __add__(self, other:velo):
	def veloadd(self, other): #other: velo
		vp = (other.lt @ self.to4d(1).T).T
		return velo(*vp[:2]/vp[3])

class vec4():
	def __init__(self,x,y,z,t):
		self.x = vec3(x,y,z)
		self.t = t if hasattr(t,"__len__") else np.repeat(t,len(x))
	
	def inframe(self,frame):
		posp = (frame.b.lt @ (self.x-frame.o).to4d(t=self.t).T).T
		return vec4(*posp)

class frame(): #here's a really cursed thought: that's basically our line-into-time class
	def __init__(self, origin, beta, t=0): #origin: vec3; beta: velo
		self.b = beta
		self.o = origin-t*beta

	def pos(self, t):
		return self.o + t*self.b

	def __neg__(self):
		pos = -(np.inv(self.b.lt) @ self.o.to4d(0).T).T
		return frame(vec3(*pos[:2]),-self.b,t=pos[3]) # i think???

	def inframe(self, frame):
		posp = vec4(*self.o,0).inframe(frame)
		return frame(posp.x,frame.b.add(self.b),posp.t)

class Thing:
	"""
	Top level class of renderable objects

	...
	Attributes
	----------
	type : int
		type of object, type constants in RayTracer.py
	pos : vec3
		vec3 position
	diffuse : vec3
		rgb vec3 colour
	mirror : float
		How much to reflect

	Methods
	-------
	diffusecolor(M)
		returns diffuse color, override for textures based on intersect position
	intersect(O, D)
		override to implement intersect for object
	light(O, D, d, scene, bounce)

	"""
	def __init__(self, type, pos, diffuse, mirror=0.5):
		"""
		type: int
			Type of object, type constants defined in RayTracer.py
		pos: vec3
			position vector
		diffuse: vec3
			diffuse color in rgb
		mirror: float, optional
			how much to reflect
		"""
		self.type = type
		self.pos = pos
		self.diffuse = diffuse
		self.mirror = mirror

	def diffusecolor(self, M):
		return self.diffuse

	def intersect(self, O, D):
		return FARAWAY

	def light(self, O, D, d, scene, bounce):
		return rgb(0, 0, 0)

class Sphere(Thing):
	"""

	"""
	def __init__(self, center, r, diffuse, mirror=0.5):
		"""

		:param center:
		:param r:
		:param diffuse:
		:param mirror:
		"""
		Thing.__init__(self, 0, center, diffuse, mirror)
		self.r = r

	def intersect(self, O, D):
		"""

		:param O:
		:param D:
		:return:
		"""
		b = 2 * D.dot(O - self.pos)
		c = abs(self.pos) + abs(O) - 2 * self.pos.dot(O) - (self.r * self.r)
		disc = (b ** 2) - (4 * c)
		sq = np.sqrt(np.maximum(0, disc))
		h0 = (-b - sq) / 2
		h1 = (-b + sq) / 2
		h = np.where((h0 > 0) & (h0 < h1), h0, h1)
		pred = (disc > 0) & (h > 0)
		return np.where(pred, h, FARAWAY)

	def light(self, O, D, d, scene, bounce):
		M = (O + D * d)  # intersection point
		N = (M - self.pos) * (1. / self.r)  # normal (numpy array)
		toL = (L - M).norm()  # direction to light
		toO = (E - M).norm()  # direction to ray origin
		nudged = M + N * .0001  # M nudged to avoid itself

		# Shadow: find if the point is shadowed or not.
		# This amounts to finding out if M can see the light
		light_distances = [s.intersect(nudged, toL) for s in scene]
		light_nearest = reduce(np.minimum, light_distances)
		seelight = light_distances[scene.index(self)] == light_nearest

		# Ambient
		color = rgb(0.05, 0.05, 0.05)

		# Lambert shading (diffuse)
		lv = np.maximum(N.dot(toL), 0)
		color += self.diffusecolor(M) * lv * seelight

		# Reflection
		if bounce < 2:
			rayD = (D - N * 2 * D.dot(N)).norm()
			color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

		# Blinn-Phong shading (specular)
		phong = N.dot((toL + toO).norm())
		color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
		return color
