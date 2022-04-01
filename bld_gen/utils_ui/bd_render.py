import os 
from datetime import datetime

ENGINE_TYPE_CYCLES = "CYCLES"
ENGINE_TYPE_EEVEE = "BLENDER_EEVEE"

class BdRender(object):
    @classmethod
    def setup_engine(cls, engine=ENGINE_TYPE_CYCLES, params=None):
        if engine == ENGINE_TYPE_CYCLES:
            cls.setup_engine_cycles(params=params)
        elif engine == ENGINE_TYPE_EEVEE:
           cls.setup_engine_eevee(params=params)
        else:
            raise ValueError("no supported {}".format(engine))

    @classmethod
    def setup_engine_cycles(cls, params=None):
        import bpy
        bpy_context = bpy.context
        scene = bpy_context.scene

        scene.render.engine = ENGINE_TYPE_CYCLES
        
        cycles = scene.cycles
        cycles.feature_set = "EXPERIMENTAL"
        cycles.device = "GPU"

        cycles.using_adaptive_sampling = True
        cycles.adaptive_thredhold = 0.2

        cycles.samples = 128
        cycles.adaptive_min_samples = 32
        cycles.time_limit = 2

        cycles.use_denoise = True

    @classmethod
    def setup_engine_eevee(cls, params=None):
        import bpy
        bpy_context = bpy.context
        scene = bpy_context.scene

        scene.render.engine = ENGINE_TYPE_EEVEE

        eevee = scene.eevee
        eevee.taa_samples = 6
        eevee.use_taa_reprojection = True

    @classmethod
    def render_scene_to_img(cls, img_pathname):
        import bpy

        s0 = datetime.now()
        if os.path.isfile(img_pathname):
            os.unlink(img_pathname)
        dir_name = os.path.dirname(img_pathname)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        bpy.context.scene.render.filepath = img_pathname
        bpy.context.scene.render.image_settings.file_format = 'PNG' # 'JPEG'
        bpy.ops.render.render(write_still=True)

        s1 = datetime.now()
        dt = s1 - s0
        print("gen {} {:.2f}sec".format(os.path.basename(img_pathname), dt.total_seconds()))

        return os.path.isfile(img_pathname)



