import taichi as ti
from vector import *
from meshtriangle  import *
from time import time
from material import *
from scene import *
from camera import Camera
import math
import random

# switch to cpu if needed
ti.init(arch=ti.gpu )
ti.set_logging_level(ti.DEBUG)

if __name__ == '__main__':

   def makeTransformations(scale,rota_angle,trans):
      scale_mat = ti.Matrix([[scale, 0, 0, 0],
                             [0, scale, 0, 0],
                             [0, 0, scale, 0],
                             [0, 0, 0, 1]])
      #绕y轴旋转
      rota_mat = ti.Matrix([[ti.cos(rota_angle), 0, ti.sin(rota_angle), 0],
                            [0, 1, 0, 0],
                            [-ti.sin(rota_angle), 0, ti.cos(rota_angle), 0],
                            [0, 0, 0, 1]])
      trans_mat2 = ti.Matrix([[1, 0, 0, trans[0]],
                              [0, 1, 0, trans[1]],
                              [0, 0, 1, trans[2]],
                              [0, 0, 0, 1]])
      return trans_mat2@rota_mat@scale_mat

   aspect_ratio = 1.0
   image_width = 784
   image_height = int(image_width / aspect_ratio)
   rays = ray.Rays(image_width, image_height)
   film_pixels = ti.Vector.field(3, dtype=float)

   ti.root.dense(ti.ij,
                 (image_width, image_height)).place(film_pixels )
   samples_per_pixel = 100
   max_depth = 10

   red = Lambert([0.65, .05, .05])
   white = Lambert([.73, .73, .73])
   green = Lambert([.12, .45, .15])
   light = Lambert_light([15, 15, 15])

   glass = Dielectric(1.5)

   moveVec=[385,125,245]
   trans= Translate( makeTransformations(180,math.pi/6,moveVec))
   spot = MeshTriangle("./models/spot/spot_triangulated_good.obj", white ,trans,"./models/spot/spot_texture.png")

   moveVec2 = [155, 25, 125]
   trans2 = Translate(makeTransformations(910, math.pi, moveVec2))
   bunny = MeshTriangle("./models/bunny/bunny.obj", white,trans2)

   left = MeshTriangle("./models/cornellbox/left.obj", red)
   right = MeshTriangle("./models/cornellbox/right.obj", green)
   floor = MeshTriangle("./models/cornellbox/floor.obj", white)
   light_ = MeshTriangle("./models/cornellbox/light.obj", light)

   shortbox  = MeshTriangle("./models/cornellbox/shortbox.obj", white)
   tallbox  = MeshTriangle("./models/cornellbox/tallbox.obj", white)

   # world

   scene = Scene()

   # scene.add(Sphere([210.0, 190.0, 290.0], 90.0, glass))
   # scene.add(Sphere([370.0, 310.0, 390.0], 90.0, white))

   scene.add(light_, 1)
   scene.add(floor)
   scene.add(left)
   scene.add(spot)
   scene.add(right)
   scene.add(bunny)
   # scene.add(shortbox)
   # scene.add(tallbox)

   scene.commit()

   # camera
   vfrom = Point([278.0, 273.0, -800.0])
   # vfrom = Point([13.0, 2.0, 3.0])
   at = Point([278.0, 273.0, 0.0])
   # at = Point([0.0, 0.0, 0.0])
   up = Vector([0.0, 1.0, 0.0])
   focus_dist =  10.0
   aperture = 0.0
   cam = Camera(vfrom, at, up, 40.0, aspect_ratio, aperture, focus_dist)



   @ti.func
   def ray_color(ray_org, ray_dir):

      col = ti.Vector([0.0, 0.0, 0.0])
      coefficient = ti.Vector([1.0, 1.0, 1.0])

      for i in range(max_depth):

         hit, p, n, front_facing, obj_index,tri_index = scene.hit_all(ray_org, ray_dir)

         if hit:
            # if index == 1:
            # print("hit",index)
            isLight, emittedCol = scene.materials.emitted(obj_index, ray_dir, p, n, front_facing)

            if isLight:  # 光源
               # if i==0:
               if front_facing:
                 col = coefficient * emittedCol
               # print("isLight", index)
               break
            else:
               reflected, out_origin, out_direction, attenuation,mat_type = scene.scatter(
                  obj_index, ray_dir, p, n, front_facing,tri_index)

               if mat_type==2:
                  coefficient *= attenuation  # 衰减
                  ray_org, ray_dir=out_origin, out_direction.normalized()
               elif front_facing:
                  pdf, ray_out_dir =scene.mix_sample(obj_index,ray_dir,p, n, front_facing)

                  if pdf>0.0 and ray_out_dir.norm()>0:

                     obj_pdf = scene.materials.sample_pdf(obj_index, ray_dir, p, n, front_facing, ray_out_dir)
                     ray_dir = ray_out_dir.normalized()

                     coefficient *=clamp(obj_pdf* attenuation/pdf,0,1)
                     ray_org=p


                  else:
                     col = ti.Vector([0.0, 0.0, 0.0])
                     break
               else:
                     col = ti.Vector([0.0, 0.0, 0.0])
                     break
         else:
            col = ti.Vector([0.0, 0.0, 0.0])
            break
      return col


   @ti.kernel
   def render():
      for i, j in film_pixels:
         col = ti.Vector.zero(float, 3)

         for k in range(samples_per_pixel):
            (u, v) = ((i + ti.random()) / image_width, (j + ti.random()) / image_height)
            ray_org, ray_dir = cam.get_ray(u, v)
            ray_dir =ray_dir.normalized()

            col += ray_color(ray_org, ray_dir)
         col /= samples_per_pixel

         # film_pixels[i, j] =  ti.sqrt(col)
         film_pixels[i, j] = clamp(ti.sqrt(col),0.0,0.999)


   t = time()
   print('starting rendering')
   render()
   print(time() - t)
   ti.imwrite(film_pixels.to_numpy(), 'out.png')
