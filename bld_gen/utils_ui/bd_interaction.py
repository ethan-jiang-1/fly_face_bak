class BdInteraction(object):
    @classmethod
    def send_dev_msg(cls, renv, msg):
        if renv is not None:
            inf_dev_msg = renv.get_interface_dev_msg()
            if inf_dev_msg is not None:
                inf_dev_msg.send_dev_msg(msg)

    @classmethod
    def refresh_screen(cls):
        import bpy
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
