import taichi as ti
from vector import *
import ray
from material import Materials
from hittable import *
from meshtriangle import *
import random
import numpy as np
from bvh import  BVHS

EPSILON=0.0001
@ti.func
def getVectors(matrix_val):
    v0 = ti.Vector([0.0,0.0,0.0])
    v1 = ti.Vector([0.0,0.0,0.0])
    v2 = ti.Vector([0.0,0.0,0.0])

    for i in ti.static(range(matrix_val.m)):
        v0[i]=matrix_val[0,i]
        v1[i]=matrix_val[1,i]
        v2[i]=matrix_val[2,i]
    return v0,v1,v2

@ti.data_oriented
class Translate:
    def __init__(self,displacement):
        self.transMatrix=displacement

    def makeTrans(self,v,f=1):
        rst = ti.Vector([v[0], v[1], v[2], f])
        rst= self.transMatrix@rst
        return  [rst[0],rst[1],rst[2]]
@ti.data_oriented
class Scene:
    def __init__(self):
        self.objects = []
        self.objlights = []
        self._light_num = 0

    def add(self, object,lighttype=0):
        object.id = len(self.objects)
        self.objects.append(object)
        self.objlights.append(lighttype)
        self._light_num+=int(lighttype==1)

    def commit(self):
        ''' Commit should be called after all objects added.
            Will compile bvh and materials. '''
        self.n = len(self.objects)
        spheres=[]
        trias=[]
        texture_datas=[]
        boxs=[]



        self.bvh = BVHS()
        self.bvh_root=self.bvh.add(self.objects)


        vexs = ti.types.matrix(n=3, m=3, dtype=ti.f32)
        texs = ti.types.matrix(n=2, m=3, dtype=ti.f32)
        tria_type = ti.types.struct(
            vs=vexs, tx=texs, normal=vexs,normal_type=ti.i32, area=ti.f32
        )
        sphe_type = ti.types.struct(
            center=vec3f, radius=ti.f32
        )
        boxs_type=ti.types.struct(
            min=vec3f,box_max=vec3f,pos=vec3i,texture_pro=vec3i,has_texture=ti.i32,id=ti.i32,area=ti.f32,geo_type=ti.i32,li_type=ti.i32
        )
        self.objs_field = boxs_type.field(shape=(self.n))
        self.materials = Materials(self.n)
        self.light_num = ti.field(ti.i32,shape=())
        self.light_num[None] = self._light_num
        triIdx=0
        texIdx=0
        sphNum=0
        # lightNum=0

        for i in range(self.n):
            obj=self.objects[i]
            bmin,bmax=obj.bounding_box
            self.objs_field[i].min=bmin
            self.objs_field[i].max=bmax
            self.objs_field[i].id=obj.id
            self.objs_field[i].area=obj.area
            self.objs_field[i].has_texture=0
            self.materials.set(i,obj.material)
            # lightNum+=self.objlights[i]
            self.objs_field[i].li_type =self.objlights[i]
            if isinstance(obj,Sphere):
                spheres.append(obj)
                self.objs_field[i].geo_type = 2
                self.objs_field[i].pos  = [sphNum,0,0]
                sphNum+=1

            if isinstance(obj,MeshTriangle):
                bvh_pos=self.bvh.add(obj.triangles)
                self.objs_field[i].pos = [triIdx, 0, bvh_pos]
                self.objs_field[i].geo_type = 1
                triIdx += obj.n
                if obj.tex_width>0:
                    self.objs_field[i].has_texture = 1
                    self.objs_field[i].texture_pro=[obj.tex_width,obj.tex_height,texIdx]
                    for w in range(obj.tex_width):
                        for h in range(obj.tex_height):
                            texture_datas.append(obj.texture_data[w,h])
                    texIdx+=obj.tex_width*obj.tex_height

                for tria in obj.triangles:
                    trias.append(tria)
        self.bvh.build()



        nSph=len(spheres)
        if nSph>0:
            self.spheres=sphe_type.field(shape=(nSph))
            for i in range(nSph):
                self.spheres[i].center = spheres[i].center
                self.spheres[i].radius = spheres[i].radius
        else:
            self.spheres = sphe_type.field(shape=(1))




        if  texIdx>0:

            self.texture_datas = ti.Vector.field(n=3, dtype=ti.i32, shape=(texIdx))
            self.texture_datas.from_numpy(np.asarray(texture_datas))
            # for i in range(texIdx):
            #     self.texture_datas[i]=texture_datas[i]
        else:
            self.texture_datas = ti.Vector.field(n=3, dtype=ti.f32, shape=(1))

        if triIdx > 0:
            self.triangles = tria_type.field(shape=(triIdx))
            for i in range(triIdx):
                preTrias=trias[i]
                self.triangles[i].vs = preTrias.vertices

                self.triangles[i].normal = preTrias.normal
                if preTrias.texcoords is not None:
                    self.triangles[i].tx = preTrias.texcoords
                self.triangles[i].normal_type = preTrias.normal_type


        del texture_datas
        del trias

    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    @ti.func
    def getTextureColor(self,obj,p,tri_index):
        tri = self.triangles[tri_index]
        tx0, tx1, tx2 = getVectors(tri.tx)
        v0, v1, v2 = getVectors(tri.vs)
        alpha, beta, gamma = Triangle.computeBarycentric2D(p.x, p.y, p.z, v0, v1, v2)
        tx = clamp(Triangle.interpolate(alpha, beta, gamma, tx0, tx1, tx2, 1.0),0,1)
        tweight=obj.texture_pro[0]
        theight=obj.texture_pro[1]
        c_idx=ti.cast(( 1-tx.y)*theight,dtype=ti.i32)*tweight+ti.cast(( tx.x)*tweight,dtype=ti.i32)-1+obj.texture_pro[2]

        color = self.texture_datas[c_idx]
        return ti.cast(color,dtype=ti.f32)/255

    @ti.func
    def hit_obj(self,obj_id,ray_origin, ray_direction, t_min,closest_so_far):
        hit_anything = False
        p = Point([0.0, 0.0, 0.0])
        n = Vector([0.0, 0.0, 0.0])
        front_facing = True
        hit_index=0
        hit_tri_index=-1
        t=0.0

        obj_box=self.objs_field[obj_id]

        if obj_box.geo_type==1:
            hit_anything,t, p, n, front_facing,hit_tri_index =self.hit_meshs(obj_box,ray_origin, ray_direction, t_min,closest_so_far)

        if obj_box.geo_type==2  :
            hit_anything,t, p, n, front_facing= self.hit_sphere(obj_box,ray_origin, ray_direction, t_min,closest_so_far)

        return hit_anything,t, p, n, front_facing,hit_tri_index
    @ti.func
    def hit_meshs(self,obj_box,ray_origin, ray_direction, t_min, t_max ):
        hit_anything = False

        closest_so_far = t_max

        p = Point([0.0, 0.0, 0.0])
        n = Vector([0.0, 0.0, 0.0])
        front_facing = True
        t=0.0
        i = 0
        hit_tri_index = -1

        curr = obj_box.pos[2]
        mat_c,mat_type,mat_roughness,mat_ior= self.get_mattype(obj_box.id)

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id,area_id = self.bvh.get_full_id(curr)

            if obj_id != -1:
                # this is a leaf node, check the sphere

                hit, _t, _p, _n, _front_facing,_hit_tri_index = self.hit_triangle(obj_id,obj_box,
                                    ray_origin, ray_direction, t_min,
                                    closest_so_far )
                # print("hit_triangle ", hit)
                valid_face=True
                if mat_type!=2 and not _front_facing:
                    valid_face=False
                if hit and valid_face:
                   hit_anything = True
                   closest_so_far = _t
                   t=_t
                   p=_p
                   n=_n
                   front_facing=_front_facing
                   hit_tri_index=_hit_tri_index

                curr = next_id
            else:
                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min,
                                     closest_so_far):
                    # add left and right children
                    # print("hit_meshs  hit_aabb" )
                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id

        return hit_anything, t, p, n, front_facing,hit_tri_index

    @ti.func
    def hit_triangle_bak1(self,tria_id,obj_box,ray_origin, ray_direction, t_min,closest_so_far ):
        tria_start=obj_box.pos[0]
        hit_tri_index=tria_start+tria_id
        obj=self.triangles[hit_tri_index]
        # n0, n1, n2 = getVectors(obj.normal)
        # n_avg=(n0+n1+n2)/3
        # if ray_direction.dot(n_avg) > 0:
        #     print("nei bu guangxian")
        # _norm=n0
        # _norm=obj.normal
        _vx=obj.vs
        # hit=ray_direction.dot(_norm) <= 0
        hit=True

        root = -1.0
        p = Point([0.0, 0.0, 0.0])
        n = Point([0.0, 0.0, 0.0])
        t = None
        front_facing = True
        v0, v1, v2 = getVectors(_vx)
        e1=v1 - v0
        e2=v2 - v0
        # print("hit1", hit)
        if hit:

            pvec = ray_direction.cross( e2)
            det = e1.dot(pvec)

            if  abs(det) < EPSILON:
                hit=False
            else:

                det_inv = 1.0 / det
                tvec = ray_origin - v0
                u = tvec.dot(pvec) * det_inv

                if (u < 0 or u > 1):
                    hit = False
                else:

                    qvec = tvec.cross( e1)
                    v = ray_direction.dot( qvec)
                    v *=det_inv

                    if (v < 0 or u + v > 1):
                        hit = False

                    else:
                        t = e2.dot(qvec) * det_inv
                        if t<=0:
                            hit = False
                        else:
                            p = ray.at(ray_origin, ray_direction, t)

                            _norm = Triangle.getNormal(obj, p)
                            # _norm = n_avg
                            # print(_norm)
                            n=_norm
                            # hit = ray_direction.dot(_norm) <= 0
                            front_facing = is_front_facing(ray_direction, n)
                            # if ray_direction.dot(_norm)>0 and obj_box.id==3:
                            #     print("nei bu guangxian",ray_origin)
                            #     print("nei bu ray_direction",ray_direction)
                            n = n if front_facing else -n

        return hit, t, p, n, front_facing,hit_tri_index

    @ti.func
    def hit_triangle(self,tria_id,obj_box,ray_origin, ray_direction, t_min,t_max ):
        tria_start=obj_box.pos[0]
        hit_tri_index=tria_start+tria_id
        obj=self.triangles[hit_tri_index]

        _vx=obj.vs

        hit=True

        root = -1.0
        p = Point([0.0, 0.0, 0.0])
        n = Point([0.0, 0.0, 0.0])
        t = None
        front_facing = True
        v0, v1, v2 = getVectors(_vx)
        e1=v1 - v0
        e2=v2 - v0
        s=ray_origin-v0
        s1=ray_direction.cross( e2)
        s2=s.cross( e1)
        det=s1.dot(e1)
        if abs(det) < EPSILON:
            hit = False
        else:
            det_inv=1.0/det
            t=s2.dot(e2)*det_inv
            if t<t_min or t_max<t:
                hit = False
            else:
                b1=s1.dot(s)*det_inv
                b2=s2.dot(ray_direction)*det_inv
                b3=1-b1-b2
                if b1<0 or b1>1 or b2<0 or b2>1 or b3<0 or b3>1:
                    hit = False
                else:
                    p = ray.at(ray_origin, ray_direction, t)
                    _norm = Triangle.getNormal(obj, p)
                    n = _norm
                    # hit = ray_direction.dot(_norm) <= 0
                    front_facing = is_front_facing(ray_direction, n)
                    n = n if front_facing else -n

        return hit, t, p, n, front_facing,hit_tri_index

    @ti.func
    def hit_sphere(self,obj_box,ray_origin, ray_direction , t_min, t_max):
        sph=self.spheres[obj_box.pos[0]]
        center=sph.center
        radius=sph.radius
        oc = ray_origin - center
        a = ray_direction.norm_sqr()
        half_b = oc.dot(ray_direction)
        c = (oc.norm_sqr() - radius ** 2)
        discriminant = (half_b ** 2) - a * c

        hit = discriminant >= 0.0
        root = -1.0
        p = Point([0.0, 0.0, 0.0])
        n = Point([0.0, 0.0, 0.0])
        t = None
        front_facing = True

        if hit:
            sqrtd = discriminant ** 0.5
            root = (-half_b - sqrtd) / a

            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a
                if root < t_min or t_max < root:
                    hit = False

        if hit:
            t=root
            p = ray.at(ray_origin, ray_direction, t)
            n = (p - center) / radius
            front_facing = is_front_facing(ray_direction, n)
            n = n if front_facing else -n

        return hit, t, p, n, front_facing

    @ti.func
    def hit_all(self, ray_origin, ray_direction):
        ''' Intersects a ray against all objects. '''
        hit_anything = False
        t_min = 0.0001
        closest_so_far = 9999999999.9
        hit_index = -1
        hit_tri_index = -1
        p = Point([0.0, 0.0, 0.0])
        n = Vector([0.0, 0.0, 0.0])
        front_facing = True
        i = 0

        # curr = self.bvh.bvh_root
        curr = self.bvh_root

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id,area_id = self.bvh.get_full_id(curr)

            if obj_id != -1:

                hit, _t, _p, _n, _front_facing,_hit_tri_index = self.hit_obj(obj_id,
                                    ray_origin, ray_direction, t_min,
                                    closest_so_far)

                if hit:
                    hit_anything = True

                    closest_so_far = _t
                    hit_index = obj_id
                    n=_n
                    p=_p
                    front_facing=_front_facing
                    hit_tri_index=_hit_tri_index

                curr = next_id
            else:
                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min,
                                     closest_so_far):
                    # add left and right children

                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id


        return hit_anything, p, n, front_facing, hit_index,hit_tri_index

    @ti.func
    def get_mattype(self, obj_index ):
        return self.materials.get(obj_index)
    @ti.func
    def scatter(self, obj_index, ray_direction, p, n, front_facing, tri_index):
        ''' Get the scattered direction for a ray hitting an object '''
        obj = self.objs_field[obj_index]
        text_color = Color([0.0, 0.0, 0.0])
        has_texture=obj.has_texture
        if has_texture==1:
            #暂时支持三角形网格的
            if obj.geo_type==1:
                text_color = self.getTextureColor(obj,p,tri_index)

        return self.materials.scatter(obj_index, ray_direction, p, n, front_facing,has_texture,text_color)

    @ti.func
    def sample_triangle(self, obj_id, triangle_pos):
        tria=self.triangles[self.objs_field[obj_id].pos[0]+triangle_pos]
        v0,v1,v2=getVectors(tria.vs)
        x = ti.sqrt(ti.random())
        y = ti.random()
        coords = v0 * (1.0  - x) + v1 * (x * (1.0 - y)) + v2 * (x * y)

        _norm=Triangle.getNormal(tria, coords)

        normal = _norm
        return coords,normal

    @ti.func
    def hit_light(self,ray_origin,  out_dir,obj):

        t_min=0.001
        t_max=infinity
        outward_normal=Vector([0.0, 1.0, 0.0])
        coords=Vector([0.0, 0.0, 0.0])
        # t = (k - p.y) / out_dir.y
        is_hit=True
        if obj.geo_type==1:
            is_hit,t, coords, outward_normal, front_facing,hit_tri_index =self.hit_meshs(obj ,ray_origin, out_dir, t_min,t_max)


        return coords,outward_normal,is_hit


    @ti.func
    def pdf_light(self,   p, n, out_dir):
        pdf = 0.0

        if self.light_num[None]>0:
           weight = 1.0 / self.light_num[None]

           for k in range(self.n):

               obj = self.objs_field[k]
               if obj.li_type == 1:
                   coords, outward_normal, is_hit_light = self.hit_light(p,  out_dir,obj)
                   if is_hit_light:
                       wo_vec = coords - p
                       pdf += weight * wo_vec.norm_sqr() / (abs(wo_vec.normalized().dot(outward_normal)) * obj.area)
           #         hit, _p, _n, front_facing, index,hit_tri_index = self.hit_all(p, out_dir)
           #         hitobj = self.objs_field[index]
           #         if hit and hitobj.li_type==1 and hitobj.id==obj.id and front_facing:
           #             wo_vec = _p - p
           #             print(_p)
           #             pdf+=weight*wo_vec.norm_sqr()/(abs(wo_vec.normalized().dot(_n))*obj.area  )
        return pdf

    @ti.func
    def sample_light(self, ray_direction, p, n, front_facing, index):
        #（多光源）随机一个光源的index
        # samplLightIdx = ti.cast(ti.random()*(self.light_num[None]-1),dtype=ti.i32)
        samplLightIdx = 0
        i=0
        coords=Point([0.0, 0.0, 0.0])
        normal=Point([0.0, 0.0, 0.0])
        color=Color([0.0, 0.0, 0.0])
        pdf=0.0
        light_obj_id=-1.0
        for k in range(self.n):

           obj = self.objs_field[k]
           if obj.li_type ==1:
               if i==samplLightIdx:
                   pos, pdf=self.bvh.sample(obj.pos[2])

                   coords, normal=self.sample_triangle(k,pos)

                   break
               else:
                   i+=1
        return coords, normal, pdf

    @ti.func
    def  mix_sample(self, index, ray_direction, p, n, front_facing):
        ray_out_dir=Vector([0.0, 0.0, 0.0])
       
        color = Color([1.0, 1.0,1.0])
        light_pdf_val = 0.0
        obj_pdf = 0.0
        pdf = 0.0

        if ti.random()<0.5:

            coords, normal, light_pdf = self.sample_light(ray_direction, p, n, front_facing, index)
            wo_vec = coords - p
            wo_dir = wo_vec.normalized()
            ray_out_dir = wo_dir
        else:
            sam_ok, obj_s_out_dir = self.materials.sample(index, ray_direction, p, n, front_facing)
            ray_out_dir =obj_s_out_dir

        light_pdf_val=self.pdf_light(p, n,ray_out_dir)
        obj_pdf = self.materials.sample_pdf(index, ray_direction, p, n, front_facing, ray_out_dir)
        pdf=0.5*light_pdf_val+0.5*obj_pdf

        return pdf,ray_out_dir