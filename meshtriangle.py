import taichi as ti
from vector import *
from loader.objloader import *
import ray
from material import Materials

import random
import numpy as np
# from bvh import BVH
from PIL import Image


@ti.func
def getVectors(matrix_val):
    # v=ti.Vector.field(n=matrix_val.n,dtype=ti.f32,shape=(3))
    # vp=[0.0,0.0]
    # if matrix_val.n==3:
    #     vp=[0.0,0.0,0.0]
    v0 = ti.Vector([0.0,0.0,0.0])
    v1 = ti.Vector([0.0,0.0,0.0])
    v2 = ti.Vector([0.0,0.0,0.0])

    for i in ti.static(range(matrix_val.m)):
        v0[i]=matrix_val[0,i]
        v1[i]=matrix_val[1,i]
        v2[i]=matrix_val[2,i]
    return v0,v1,v2

@ti.data_oriented
class Triangle:
    def __init__(self,vexs,   _norms,   _texcoords,  material):
        v0=ti.Vector(vexs[0])
        v1=ti.Vector(vexs[1])
        v2=ti.Vector(vexs[2])
        self.vertices = vexs
        self.texcoords =None
        self.material=material

        self.id = -1
        e1 = v1 - v0
        e2 = v2 - v0
        self.area = e1.cross(e2).norm() * 0.5

        # n = e1.cross(e2).normalized()
        # _n = [n[0], n[1], n[2]]
        # self.normal = ti.Matrix([_n, _n, _n])
        # # self.normal = n
        # self.normal_type = 1
        if _texcoords is not None and len(_texcoords) == 3:
            self.texcoords = _texcoords
        if _norms is None or len(_norms)==0:
          n = e1.cross(e2).normalized()
          _n=[n[0],n[1],n[2]]
          self.normal =ti.Matrix([_n,_n,_n])
          # self.normal = n
          self.normal_type = 1

        else:
            self.normal =ti.Matrix(_norms)
            self.normal_type = 3
        # print(self.normal)
        self.box_min = [min(v0[0],v1[0],v2[0],),min(v0[1],v1[1],v2[1],),min(v0[2],v1[2],v2[2],) ]
        self.box_max = [max(v0[0],v1[0],v2[0],),max(v0[1],v1[1],v2[1],),max(v0[2],v1[2],v2[2],)]
        self.box_center = [0.5 * self.box_min[0] + 0.5 * self.box_max[0], 0.5 * self.box_min[1] + 0.5 * self.box_max[1],
                       0.5 * self.box_min[2] + 0.5 * self.box_max[2]]
        # self.box = bounds3(box_min,box_max)
        # print(e1)
        # print(_v0)
        # print("Triangle aabb min",self.box_min)
        # print("Triangle aabb max",self.box_max)


    @property
    def bounding_box(self):
        # return self.box
        return self.box_min, self.box_max

    @property
    def center(self):
        # return self.box
        return self.box_center

    @ti.func
    def sample(self):
        x = ti.sqrt(ti.random())
        y = ti.random()
        v0,v1,v2=getVectors(self.vertices)
        pos = v0 * (1.0 - x) + v1 * (x * (1.0 - y)) + v2 * (x * y)
        return pos,self.normal

    @staticmethod
    @ti.func
    def computeBarycentric2D(x, y,z, v0, v1, v2):
        # e1=ti.Vector([v1.x - v2.x,v1.y - v2.y,v1.z - v2.z])
        # e2=ti.Vector([v0.x - v2.x,v0.y - v2.y,v0.z - v2.z])
        # area=e1.cross(e2).norm()
        # a1=ti.Vector([ x - v2.x, y - v2.y, z - v2.z]).cross(e1).norm()
        # a2=ti.Vector([ x - v2.x, y - v2.y, z - v2.z]).cross(e2).norm()
        # c1=a1/area
        # c2=a2/area
        # c3=1-c1-c2
        c1 = (x * (v1.y - v2.y) + (v2.x - v1.x) * y + v1.x * v2.y - v2.x * v1.y) / (
                v0.x * (v1.y - v2.y) + (v2.x - v1.x) * v0.y + v1.x * v2.y - v2.x * v1.y)
        c2 = (x * (v2.y - v0.y) + (v0.x - v2.x) * y + v2.x * v0.y - v0.x * v2.y) / (
                v1.x * (v2.y - v0.y) + (v0.x - v2.x) * v1.y + v2.x * v0.y - v0.x *  v2.y)
        c3 = (x * (v0.y - v1.y) + (v1.x - v0.x) * y + v0.x * v1.y - v1.x * v0.y) / (
                v2.x * (v0.y - v1.y) + (v1.x - v0.x) * v2.y + v0.x * v1.y - v1.x * v0.y)
        return c1, c2, c3

    @staticmethod
    @ti.func
    def getNormal(tria_field,p):
        n0, n1, n2 =getVectors(tria_field.normal)
        _normal=n0

        if tria_field.normal_type==3:
            v0, v1, v2 = getVectors(tria_field.vs)
            alpha, beta, gamma = Triangle.computeBarycentric2D(p.x, p.y, p.z, v0, v1, v2)
            _normal = Triangle.interpolate(alpha, beta, gamma, n0 , n1 , n2 , 1.0).normalized()
        return _normal

    @staticmethod
    @ti.func
    def interpolate( alpha, beta, gamma, vert1, vert2, vert3, weight):
        return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight


    @staticmethod
    @ti.func
    def makeInterpolateVal(x, y, v0, v1, v2, weight):
        alpha,beta,gamma = Triangle.computeBarycentric2D(x, y, v0, v1, v2)
        return Triangle.interpolate(alpha,beta,gamma, v0, v1, v2, weight)




@ti.data_oriented
class MeshTriangle:
    def __init__(self,filename, mat,trans=None,texture_filename=None):
        self.triangles = []
        self.trans = trans
        self.material =mat
        self.tex_width =-1
        self.tex_height =-1
        self.id=-1
        objLoader = OBJ( filename)
        assert(len(objLoader.faces)>0)
        if texture_filename is not None:
            im = Image.open(texture_filename)
            self.texture_data = np.array(im)
            self.tex_width = self.texture_data.shape[0]
            self.tex_height = self.texture_data.shape[1]
        vexs = objLoader.vertices
        self.box_min = [infinity,infinity,infinity]
        self.box_max = [-infinity,-infinity,-infinity]
        for face, norms, texcoords, material in objLoader.faces:
            # print("texcoords",texcoords )
            # print("material",material )
            face_vertices=[]
            face_norms=[]
            face_texcoords=[]
            for v in face:

                vert = vexs[v-1]
                if trans is not None:
                    vert = trans.makeTrans(vert)
                face_vertices.append(vert)

                self.box_min = [min(self.box_min[0], vert[0]), min(self.box_min[1], vert[1]), min(self.box_min[2], vert[2])]
                self.box_max = [max(self.box_max[0], vert[0]), max(self.box_max[1], vert[1]), max(self.box_max[2], vert[2])]
            for v in norms:
                if v>0:
                    _n=objLoader.normals[v-1]
                    if trans is not None:
                        _n = trans.makeTrans(_n,0)
                    face_norms.append(_n)
            for v in texcoords:
                if v>0:
                    face_texcoords.append(objLoader.texcoords[v-1])
            if material is None or len(material)==0:
                material=self.material
            tria =Triangle(face_vertices,face_norms,face_texcoords,material)
            tria.id=len(self.triangles)
            self.triangles.append(tria)
        self.box_center = [0.5 * self.box_min[0] + 0.5 * self.box_max[0], 0.5 * self.box_min[1] + 0.5 * self.box_max[1],
                           0.5 * self.box_min[2] + 0.5 * self.box_max[2]]
        print("box_min",self.box_min)
        print("box_max",self.box_max)
        # print("filename",filename)
        self.n = len(self.triangles)

        _area=0.0
        for tri in self.triangles:

            # print(tri.area)
            _area +=tri.area
            # _area +=tri.getArea()
        self.area  = _area


    @property
    def bounding_box(self ):
        return self.box_min, self.box_max

    @property
    def center(self):
        # return self.box
        return self.box_center

    @ti.func
    def sample(self):
        return self.bvh.sample()


