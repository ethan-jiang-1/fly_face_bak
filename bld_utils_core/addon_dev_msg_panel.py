import bpy 

bl_info = {
    "name": "DevMsgPanel",
}

MAX_MSG_NUM = 20

class MY_PT_DevMsgPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DevMsgPanel"
    bl_options = {"DEFAULT_CLOSED"}

    bl_idname = "MYADDON_PT_MsgPanelId"
    bl_label = "DevMsg"

    g_msgs = []
    g_region_refresh = None

    def draw(self, context):
        layout = self.layout
        
        for msg in self.g_msgs:
            layout.label(text=msg, text_ctxt="msg")

    @classmethod
    def receive_dev_msg(cls, sender, msg):
        if len(cls.g_msgs) > 0:
            if cls.g_msgs[-1] == msg:
                return
        
        cls.g_msgs.append(msg)
        if len(cls.g_msgs) > MAX_MSG_NUM:
            cls.g_msgs.pop(0)
        
        cls._ensure_refresh_region()
        if cls.g_region_refresh is not None:
            cls.g_region_refresh.tag_redraw()

    @classmethod
    def init(cls):
        cls.g_msgs.clear()
        cls._ensure_refresh_region()
        if cls.g_region_refresh is not None:
            cls.g_region_refresh.tag_redraw()

    @classmethod
    def _ensure_refresh_region(cls):
        from bld_utils_ui.wnd_finder import WndFinder
        if cls.g_region_refresh is None:
            region = WndFinder.find_wnd_region(area_type="VIEW_3D", region_type="UI")
            print("find refresh region", region)
            cls.g_region_refresh = region
        
  
def register():
    print("*register DevMsgPanel")
    try:
        bpy.utils.register_class(MY_PT_DevMsgPanel)
        return True
    except Exception as ex:
        print("Exception occured", ex)
    return False

def unregister():
    print("*unregister DevMsgPanel")
    try:
        bpy.utils.unregister_class(MY_PT_DevMsgPanel)
        return True
    except Exception as ex:
        print("Exception occured", ex)
    return False
