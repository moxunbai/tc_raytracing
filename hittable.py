import taichi as ti
from vector import *
import ray
from material import Materials
import random
import numpy as np
# from bvh import BVH


@ti.func
def is_front_facing(ray_direction, normal):
    return ray_direction.dot(normal) < 0.0
 
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
        self.area=4*math.pi*radius*radius
        self.id = -1
        self.box_min = [
            self.center[0] - radius, self.center[1] - radius,
            self.center[2] - radius
        ]
        self.box_max = [
            self.center[0] + radius, self.center[1] + radius,
            self.center[2] + radius
        ]

    @property
    def bounding_box(self):
        return self.box_min, self.box_max


BRANCH = 1.0
LEAF = 0.0
