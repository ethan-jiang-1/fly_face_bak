import bpy 

class PanelFinder():
    @classmethod
    def list_all_panel_klass(cls):
        for idx, panel in enumerate(bpy.types.Panel.__subclasses__()):
            print(idx, panel, panel.__class__)

    @classmethod
    def has_panel_klass_registered(cls, clsname):
        for panel in bpy.types.Panel.__subclasses__():
            if str(panel).find(clsname):
                print("{} has installed {}".format(clsname, panel))
                return True
        return False
