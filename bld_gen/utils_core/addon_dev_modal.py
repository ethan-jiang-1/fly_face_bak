import bpy 

bl_info = {
    "name": "DevModal",
}

class MY_OT_DevModal(bpy.types.Operator):
    bl_label = "DevModal"
    bl_idname = "ui.dev_modal"
    bl_options = {'REGISTER'}  # , 'UNDO'}

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
    print("*register DevModal")
    try:
        bpy.utils.register_class(MY_OT_DevModal)
        return True
    except Exception as ex:
        print("Exception occured", ex)
    return False

def unregister():
    print("*unregister DevModal")
    try:
        bpy.utils.unregister_class(MY_OT_DevModal)
        return True
    except Exception as ex:
        print("Exception occured", ex)
    return False


def test_ui():
    print("bring confirmation")

    bpy.ops.ui.dev_modal('INVOKE_DEFAULT')

    print("end confirmation")
