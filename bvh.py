import taichi as ti
import copy
import random
from vector import *

def surrounding_box(box1, box2):
    ''' Calculates the surround bbox of two bboxes '''
    box1_min, box1_max = box1
    box2_min, box2_max = box2

    small = [
        min(box1_min[0], box2_min[0]),
        min(box1_min[1], box2_min[1]),
        min(box1_min[2], box2_min[2])
    ]
    big = [
        max(box1_max[0], box2_max[0]),
        max(box1_max[1], box2_max[1]),
        max(box1_max[2], box2_max[2])
    ]
    return small, big


def sort_obj_list(obj_list):
    ''' Sort the list of objects along the longest directional span '''
    def get_x(e):
        # return e.center[0]
        return e.center[0]

    def get_y(e):
        # return e.center[1]
        return e.center[1]

    def get_z(e):
        # return e.center[2]
        return e.center[2]

    def box_compare(a, b, axis):
        box1_min, box1_max = a.bounding_box
        box2_min, box2_max = b.bounding_box

        if box1_min.e[axis] < box2_min.min().e[axis]:
            return 1
        elif box1_min.e[axis] > box2_min.min().e[axis]:
            return -1
        else:
            return 0
    bounds =[ obj.bounding_box  for obj in obj_list]

    box_min = [
        min([bound[0][0] for bound in bounds]),
        min([bound[0][1] for bound in bounds]),
        min([bound[0][2] for bound in bounds])
    ]
    box_max = [
        min([bound[1][0] for bound in bounds]),
        min([bound[1][1] for bound in bounds]),
        min([bound[1][2] for bound in bounds])
    ]
    span_x, span_y, span_z = (box_max[0] - box_min[0],
                              box_max[1] - box_min[1],
                              box_max[2] - box_min[2])
    if span_x >= span_y and span_x >= span_z:
        obj_list.sort(key=get_x)
    elif span_y >= span_z:
        obj_list.sort(key=get_y)
    else:
        obj_list.sort(key=get_z)
    return obj_list


class BVHNode:
    ''' A bvh node for constructing the bvh tree.  Note this is done on CPU '''

    left = None
    right = None
    obj = None
    box_min = box_max = []
    id = 0
    parent = None
    total = 0
    area = 0

    def __init__(self, object_list, parent):
        self.parent = parent
        obj_list = copy.copy(object_list)

        span = len(object_list)
        if span == 1:
            # one obj, set to sphere bbox
            self.obj = obj_list[0]
            self.box_min, self.box_max = obj_list[0].bounding_box
            self.total = 1
            self.area = self.obj.area
        else:
            # set left and right child and this bbox is the sum of two
            sorted_list = sort_obj_list(obj_list)
            mid = int(span / 2)
            self.left = BVHNode(obj_list[:mid], self)
            self.right = BVHNode(obj_list[mid:], self)
            self.box_min, self.box_max = surrounding_box(
                self.left.bounding_box, self.right.bounding_box)
            self.total = self.left.total + self.right.total + 1
            self.area = self.left.area + self.right.area

    @property
    def bounding_box(self):
        return self.box_min, self.box_max

    @property
    def next(self):
        ''' Returns the next node to walk '''
        node = self

        while True:
            if node.parent is not None and node.parent.right is not node:
                return node.parent.right
            elif node.parent is None:
                return None
            else:
                node = node.parent
        return None


@ti.data_oriented
class BVHS:

    root_list=[]
    # max_total=0
    total_count=0
    def __init__(self):
        pass

    def add(self,object_list):
        root = BVHNode(object_list, None)
        pos=self.total_count
        self.root_list.append(root)
        self.total_count+=root.total

        return pos

    def build(self):
        n_root=len(self.root_list)
        self.bvh_obj_id = ti.field(ti.i32)
        self.bvh_left_id = ti.field(ti.i32)
        self.bvh_right_id = ti.field(ti.i32)
        self.bvh_next_id = ti.field(ti.i32)
        self.bvh_area = ti.field(ti.f32)
        self.bvh_min = ti.Vector.field(3, dtype=ti.f32)
        self.bvh_max = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, self.total_count).place(self.bvh_obj_id, self.bvh_left_id,
                                         self.bvh_right_id, self.bvh_next_id,
                                         self.bvh_min, self.bvh_max, self.bvh_area)
        ''' building function. Compress the object list to structure'''


        # first walk tree and give ids
        def do_walk_bvh(node,start_pos):
           i = 0
           def walk_bvh(node):
               nonlocal i
               node.id = i+start_pos

               i += 1
               if node.left:
                   walk_bvh(node.left)
               if node.right:
                   walk_bvh(node.right)
           walk_bvh(node)

        start_pos=0
        for per_root in self.root_list:
            do_walk_bvh(per_root,start_pos)
            start_pos+=per_root.total

        def save_bvh(node ):
            id = node.id

            self.bvh_obj_id[id] = node.obj.id if node.obj is not None else -1
            self.bvh_left_id[
                id] = node.left.id if node.left is not None else -1
            self.bvh_right_id[
                id] = node.right.id if node.right is not None else -1
            self.bvh_next_id[
                id] = node.next.id if node.next is not None else -1
            self.bvh_min[id] = node.box_min
            self.bvh_max[id] = node.box_max
            self.bvh_area[id] = node.area

            if node.left is not None:
                save_bvh(node.left )
            if node.right is not None:
                save_bvh(node.right )
        s_pos=0
        for i in range(n_root):
            per_root=self.root_list[i]
            # print("save_bvh rootid:", per_root.id)
            save_bvh(per_root )
            # s_pos+=per_root.total



    @ti.func
    def hit(self, ray_origin, ray_direction):
        print(1)
    @ti.func
    def get_id(self, bvh_id):
        ''' Get the obj id for a bvh node '''
        return self.bvh_obj_id[bvh_id]

    @staticmethod
    @ti.func
    def getdival(a,b):
        v=0.0
        if a==0:
            v=0.0
        else:
          if b==0:
            v=a*infinity
          else:
            v=a/b
        return v
    @ti.func
    def hit_aabb2(self, bvh_id, ray_origin, ray_direction, t_min, t_max):
        o = ray_origin
        pMin = self.bvh_min[bvh_id]
        pMax = self.bvh_max[bvh_id]
        tminx = BVHS.getdival(pMin.x - o.x,ray_direction.x)
        tminy = BVHS.getdival(pMin.y - o.y,ray_direction.y)
        tminz = BVHS.getdival(pMin.z - o.z,ray_direction.z)
        tmaxx = BVHS.getdival(pMax.x - o.x,ray_direction.x)
        tmaxy = BVHS.getdival(pMax.y - o.y,ray_direction.y)
        tmaxz = BVHS.getdival(pMax.z - o.z,ray_direction.z)

        if ray_direction.x<=0:
            tminx, tmaxx= tmaxx,tminx
        if ray_direction.y<=0:
            tminy, tmaxy=tmaxy,tminy
        if ray_direction.z<=0:
            tminz, tmaxz=tmaxz,tminz

        tenter = ti.max(tminx, ti.max(tminy, tminz))

        texit = ti.min(tmaxx, ti.min(tmaxy, tmaxz))
        # print("texit", texit)
        # print("tenter", tenter)
        # if bvh_id>=3:
        #     print("texit",texit)
        #     print("tenter",tenter)
        return texit > 0 and texit >= tenter
    @ti.func
    def hit_aabb(self, bvh_id, ray_origin, ray_direction, t_min, t_max):
        ''' Use the slab method to do aabb test'''
        intersect = 1
        min_aabb = self.bvh_min[bvh_id]
        max_aabb = self.bvh_max[bvh_id]

        for i in ti.static(range(3)):
            if ray_direction[i] == 0:
                if ray_origin[i] < min_aabb[i] or ray_origin[i] > max_aabb[i]:

                    intersect = 0
            else:
                i1 = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
                i2 = (max_aabb[i] - ray_origin[i]) / ray_direction[i]

                new_t_max = ti.max(i1, i2)
                new_t_min = ti.min(i1, i2)

                t_max = ti.min(new_t_max, t_max)
                t_min = ti.max(new_t_min, t_min)

        if t_min > t_max:
            intersect = 0
        return intersect

    @ti.func
    def get_full_id(self, i):
        ''' Gets the obj id, left_id, right_id, next_id for a bvh node '''
        return self.bvh_obj_id[i], self.bvh_left_id[i], self.bvh_right_id[
            i], self.bvh_next_id[i], self.bvh_area[i]

    @ti.func
    def sample(self,bvh_pos):

        curr = bvh_pos
        obj_id, left_id, right_id, next_id, area = self.get_full_id(curr)
        p = ti.random() * area
        pdf=1.0/area
        pos=0.0
        # normal=ti.Vector([0.0,0.0,0.0])

        while curr != -1:
            # obj_id, left_id, right_id, next_id = self.bvh.get_full_id(curr)
            if  obj_id != -1:
                pos =obj_id
                break
            else:
                # obj_id, left_id, right_id, next_id, area = self.bvh.get_full_id(left_id)
                if p<area:
                    curr=left_id
                else:
                    curr = right_id
                    p=p-area
                obj_id, left_id, right_id, next_id, area = self.get_full_id(curr)

        return pos,pdf