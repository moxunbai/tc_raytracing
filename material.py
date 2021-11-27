import taichi as ti
# from taichi_glsl.vector import reflect
from vector import *


@ti.func
def reflectance(cosine, idx):
    r0 = ((1.0 - idx) / (1.0 + idx))**2
    return r0 + (1.0 - r0) * ((1.0 - cosine)**5)


@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n


@ti.func
def refract(v, n, etai_over_etat):
    cos_theta = min(-v.dot(n), 1.0)
    r_out_perp = etai_over_etat * (v + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.norm_sqr())) * n
    return r_out_perp + r_out_parallel

@ti.func
def toWord(rn, n ):
    w=n.normalized()
    a=Vector([0, 1, 0]) if abs(w.x> 0.9) else Vector([1, 0, 0])
    v= w.cross(a).normalized()
    u=w.cross(v)
    return rn.x*u+rn.y*v+rn.z*w



class _material:
    def scatter(self, in_direction, p, n):
        pass


class Lambert(_material):
    def __init__(self, color):
        self.color = color
        self.index = 0
        self.roughness = 0.0
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color,has_texture,text_color):
        # out_direction = n + random_in_hemisphere(n)
        attenuation = color
        if has_texture==1:

            attenuation= text_color
        return True, p, Vector([0.0, 0.0, 0.0]), attenuation
        # return  None,attenuation
    @staticmethod
    @ti.func
    def sample(in_direction, p, n ):
        # out_direction = n + random_in_hemisphere(n)

        x_1 = ti.random()
        x_2 = ti.random()

        phi = 2*math.pi * x_2

        x=ti.cos(phi)*ti.sqrt(x_1)
        y=ti.sin(phi)*ti.sqrt(x_1)
        z=ti.sqrt(1-x_1)

        a=Vector([x , y , z ])
        out_direction =toWord(a,n).normalized()

        # out_direction=
        # if n.dot(out_direction)<=0:
        #     out_direction=-out_direction
        return  True,out_direction


    @staticmethod
    @ti.func
    def sample_pdf(in_dir , p, n,out_dir):

        cosine = n.dot(out_dir)
        pdf =0.0 if(cosine <= 0) else (cosine / math.pi)
        return  pdf


class Metal(_material):
    def __init__(self, color, roughness):
        self.color = color
        self.index = 1
        self.roughness = min(roughness, 1.0)
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color, roughness,has_texture,text_color):
        # out_direction = reflect(in_direction.normalized(),
        #                         n) + roughness * random_in_unit_sphere()
        attenuation = color
        # reflected = out_direction.dot(n) > 0.0
        return True, p, Vector([0.0, 0.0, 0.0]), attenuation
        # return  None,attenuation
    @staticmethod
    @ti.func
    def sample(in_direction, p, n ):

        return  False,Vector([0.0, 0.0, 0.0])

    @staticmethod
    @ti.func
    def sample_pdf(in_dir, p, n, out_dir):
        return 0.0

class Dielectric(_material):
    def __init__(self, ior):
        self.color = Color(1.0, 1.0, 1.0)
        self.index = 2
        self.roughness = 0.0
        self.ior = ior

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color, ior, front_facing,has_texture,text_color):
        refraction_ratio = 1.0 / ior if front_facing else ior
        unit_dir = in_direction.normalized()
        cos_theta = min(-unit_dir.dot(n), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        out_direction = Vector([0.0, 0.0, 0.0])
        cannot_refract = refraction_ratio * sin_theta > 1.0
        if cannot_refract or reflectance(cos_theta,
                                         refraction_ratio) > ti.random():
            out_direction = reflect(unit_dir, n)
        else:
            out_direction = refract(unit_dir, n, refraction_ratio)
        attenuation = color

        return  True,p,out_direction, attenuation

    @staticmethod
    @ti.func
    def sample(in_direction, p, n ):

        return False, None

class Lambert_light(_material):
    def __init__(self, color):
        self.color = color
        self.index = 10
        self.roughness = 0.0
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color):
        pass

    @staticmethod
    @ti.func
    def sample(in_direction, p, n, color):
        pass

@ti.data_oriented
class Materials:
    ''' List of materials for a scene.'''
    def __init__(self, n):
        self.roughness = ti.field(ti.f32)
        self.colors = ti.Vector.field(3, dtype=ti.f32)
        self.mat_index = ti.field(ti.u32)
        self.ior = ti.field(ti.f32)
        ti.root.dense(ti.i, n).place(self.roughness, self.colors,
                                     self.mat_index, self.ior)

    def set(self, i, material):
        self.colors[i] = material.color
        self.mat_index[i] = material.index
        self.roughness[i] = material.roughness
        self.ior[i] = material.ior

    @ti.func
    def get(self, i ):
        return self.colors[i],self.mat_index[i] ,self.roughness[i] ,self.ior[i]

    @ti.func
    def scatter(self, i, ray_direction, p, n, front_facing,has_texture,text_color):
        ''' Get the scattered ray that hits a material '''
        mat_index = self.mat_index[i]
        color = self.colors[i]
        roughness = self.roughness[i]
        ior = self.ior[i]
        reflected = True
        out_origin = Point([0.0, 0.0, 0.0])
        out_direction = Vector([0.0, 0.0, 0.0])
        attenuation = Color([0.0, 0.0, 0.0])

        if mat_index == 0:
            reflected, out_origin, out_direction, attenuation = Lambert.scatter(
                ray_direction, p, n, color,has_texture,text_color)
            # out_direction, attenuation = Lambert.scatter(
            #     ray_direction, p, n, color)
        elif mat_index == 1:
            reflected, out_origin, out_direction, attenuation = Metal.scatter(
                ray_direction, p, n, color, roughness,has_texture,text_color)
            # out_direction, attenuation = Metal.scatter(
            #     ray_direction, p, n, color, roughness)
        elif mat_index == 2:
            reflected, out_origin, out_direction, attenuation = Dielectric.scatter(
                ray_direction, p, n, color, ior, front_facing,has_texture,text_color)
            # out_direction, attenuation = Dielectric.scatter(
            #     ray_direction, p, n, color, ior, front_facing)
        elif mat_index == 10:
            reflected=False

        return reflected, out_origin, out_direction, attenuation,mat_index
        # return  out_direction, attenuation,mat_index

    @ti.func
    def emitted(self, i, ray_direction, p, n, front_facing):
        mat_index = self.mat_index[i]
        color = self.colors[i]

        isLight=mat_index==10
        return isLight,color
    @ti.func
    def sample(self, i, ray_direction, p, n, front_facing):
        mat_index = self.mat_index[i]
        color = self.colors[i]
        roughness = self.roughness[i]
        ior = self.ior[i]
        result_ok = False
        out_direction = Vector([0.0, 0.0, 0.0])


        if mat_index == 0:

            result_ok,out_direction  = Lambert.sample(
                ray_direction, p, n )
        elif mat_index == 1:

            result_ok,out_direction = Metal.sample(
                ray_direction, p, n )

        return result_ok,out_direction

    @ti.func
    def sample_pdf(self, i, ray_direction, p, n, front_facing,ray_out_dir):
        mat_index = self.mat_index[i]
        color = self.colors[i]
        roughness = self.roughness[i]
        ior = self.ior[i]

        pdf = 0.0

        if mat_index == 0:

            pdf  = Lambert.sample_pdf(
                ray_direction, p, n, ray_out_dir)
        elif mat_index == 1:

            pdf = Metal.sample_pdf(
                ray_direction, p, n, ray_out_dir)


        return pdf
