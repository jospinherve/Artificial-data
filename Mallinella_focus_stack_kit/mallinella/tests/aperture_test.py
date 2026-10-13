import bpy, numpy as np, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene; s.unit_settings.system='METRIC'; s.unit_settings.scale_length=float(sys.argv[-1])
s.render.engine='CYCLES'; s.cycles.samples=256; s.cycles.use_denoising=False
s.render.resolution_x=1000; s.render.resolution_y=1000; s.render.film_transparent=False
w=bpy.data.worlds.new('w'); s.world=w; w.color=(0,0,0)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0,250,0))
m=bpy.data.materials.new('e'); m.use_nodes=True; nt=m.node_tree; nt.nodes.clear()
e=nt.nodes.new('ShaderNodeEmission'); e.inputs[1].default_value=50; o=nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0],o.inputs[0])
bpy.context.object.data.materials.append(m)
cam=bpy.data.cameras.new('c'); cam.lens=100; cam.sensor_width=36; cam.sensor_fit='HORIZONTAL'; cam.clip_start=1; cam.clip_end=1000
cam.dof.use_dof=True; cam.dof.focus_distance=200; cam.dof.aperture_fstop=100e-3/10.0; cam.dof.aperture_blades=0
co=bpy.data.objects.new('c',cam); s.collection.objects.link(co); s.camera=co; co.rotation_euler=(1.5707963,0,0)
s.render.image_settings.file_format='OPEN_EXR'; s.render.filepath='/tmp/claude-0/-home-claude/19e49ecb-12d8-53a4-9a86-c91bd3333988/scratchpad/ap.exr'
s.view_settings.view_transform='Standard'
bpy.ops.render.render(write_still=True)
im=bpy.data.images.load(s.render.filepath); a=np.array(im.pixels[:]).reshape(1000,1000,4)[...,0]
row=a[500]; thr=a.max()*0.1
print("RESULT scale",sys.argv[-1],"blur width px", (a.max(axis=0)>thr).sum(), "expected 27.8 if aperture in scene units")
