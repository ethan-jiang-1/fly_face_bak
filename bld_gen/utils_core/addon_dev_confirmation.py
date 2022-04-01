import bpy 

bl_info = {
    "name": "DevConfirmation",
}

class MY_OT_DevConfirmation(bpy.types.Operator):
    bl_label = "DevConfirmation"
    bl_idname = "ui.dev_confirmation"
    bl_options = {'REGISTER'}  # , 'UNDO'}

    name : bpy.props.StringProperty()
    first_mouse_x : bpy.props.IntProperty()
    first_value : bpy.props.FloatProperty()

    #@classmethod
    #def poll(cls, context):
    #    return True

    def draw(self, context):
        layout = self.layout
        layout.label("confirmed?")
        return "{FINISHED}"

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            #delta = self.first_mouse_x - event.mouse_x
            #context.object.location.x = self.first_value + delta * 0.01
            pass
        elif event.type == 'LEFTMOUSE':
            #self.report({'INFO'}, "Modal: Finished")
            #return {'FINISHED'}
            pass
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            #context.object.location.x = self.first_value
            self.report({'INFO'}, "Modal: Cancelled")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.object:
            #self.first_mouse_x = event.mouse_x
            #self.first_value = context.object.location.x
            self.report({'INFO'}, "Modal: Enter modal mode")
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, no need to enter modal")
            return {'CANCELLED'}

  
def register():
    print("*register DevConfirmation")
    try:
        bpy.utils.register_class(MY_OT_DevConfirmation)
        return True
    except Exception as ex:
        print("Exception occured", ex)
    return False

def unregister():
    print("*unregister DevConfirmation")
    try:
        bpy.utils.unregister_class(MY_OT_DevConfirmation)
        return True
    except Exception as ex:
        print("Exception occured", ex)
    return False


def test_ui():
    print("bring confirmation")
    #bpy.ops.ui.dev_confirmation()

    bpy.ops.ui.dev_confirmation('INVOKE_DEFAULT')

    #wm = bpy.context.window_manager
    #wm.invoke_props_popup(MY_OT_DevConfirmation)
    print("end confirmation")
