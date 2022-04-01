import os
import sys
import importlib 

# POSSIBLE PLAY MODE: NATIVE,  VRM, ARKIT, FACS
ENV_NEW = {"PLAY_MODE": "NATIVE|VRM|ARKIT",
           "BS_STRENGTH": "0.0"}
RELOAD_ALL_MODULES = True 

def _handle_dev_msg(sender, **kwargs):
    print("*hm:", sender, kwargs)
    from bld_gen.utils_core.addon_dev_msg_panel import MY_PT_DevMsgPanel
    MY_PT_DevMsgPanel.receive_dev_msg(sender, kwargs["message"])

class _InterfaceDevMsg(object):
    def __init__(self):
        from bld_gen.utils_core.dev_signal_msg import get_sig_msg
        self.sig_msg = get_sig_msg()

        from bld_gen.utils_core.addon_dev_msg_panel import MY_PT_DevMsgPanel
        MY_PT_DevMsgPanel.init()

    def send_dev_msg(self, msg):
        if self.sig_msg is not None:
            ret = self.sig_msg.send('sender', message=msg)
            return ret
        return []

    def connect_dev_msg(self, msg_handler):
        if self.sig_msg is not None:
            self.sig_msg.connect(msg_handler)

    def disconnect_dev_msg(self, msg_hander):
        if self.sig_msg is not None:
            self.sig_msg.disconnect(msg_hander)

class _RunEnv(object):
    instance_dev_msg = None

    @classmethod
    def init(cls):
        if cls.instance_dev_msg is None:
            root_dir = cls.get_root_dir()
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            cls.instance_dev_msg = _InterfaceDevMsg()

    @classmethod
    def get_interface_dev_msg(cls):
        return cls.instance_dev_msg

    @classmethod
    def _pop_last_sub_folder(cls, root_dir):
        subfolders = ["_reserved_", "bldmc_"]
        names = root_dir.split(os.sep)
        for subfolder in subfolders:
            if names[-1].startswith(subfolder):
                names.pop()
        root_dir = os.sep.join(names)
        return root_dir

    @classmethod
    def get_root_dir(cls):
        host_file = __file__
        print("host_file", host_file)
        host_dir = os.path.dirname(host_file)
        if host_dir.endswith(".blend"):
            root_dir = os.path.dirname(host_dir)
        else:
            raise ValueError("blend need to be open first")
        root_dir = cls._pop_last_sub_folder(root_dir)
        root_dir = os.path.dirname(root_dir)
        print("root_dir", root_dir)
        return root_dir

def _update_sys_paths(renv):
    import bpy

    blend_dir = os.path.dirname(bpy.data.filepath)
    if blend_dir not in sys.path:
        sys.path.append(blend_dir)

    root_dir = renv.get_root_dir()
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    print("updated sys.path")
    for path in sys.path:
        print(path)
    print()

def _prepare_run_env():
    import os 
    for key, val in ENV_NEW.items():
        os.environ[key] = val

    _RunEnv.init()

    inf_dev_msg = _RunEnv.get_interface_dev_msg()
    if inf_dev_msg is not None:
        inf_dev_msg.connect_dev_msg(_handle_dev_msg)

    return _RunEnv

def _prepare_to_run(renv):
    print("prepare to run")
    from bld_gen.utils_core.addon_dev_msg_panel import register
    register()

def _prepare_to_exit(renv):
    print("prepare to exit")
    #from bld_gen.utils_core.dev_msg_panel import unregister
    #unregister()
    inf_dev_msg = renv.get_interface_dev_msg()
    if inf_dev_msg is not None:
        inf_dev_msg.disconnect_dev_msg(_handle_dev_msg)


def _dyn_load_to_run(renv):
    import bld_gen.model_inspect.shapekeys_inspector as shapekeys_inspector
    print("reload inspect_blendshape", shapekeys_inspector)
    importlib.reload(shapekeys_inspector)
    shapekeys_inspector.do_inspect(renv)

def _reload_all_modules(renv):
    import bld_gen.utils_core.reload_all_modules as reload_all_modules
    print("reload reload_all_modules", reload_all_modules)
    importlib.reload(reload_all_modules)
    root_dir = renv.get_root_dir()
    reload_all_modules.reload_all_modules(root_dir)

def debug_main(renv):
    _prepare_to_run(renv)

    _dyn_load_to_run(renv)
    
    _prepare_to_exit(renv)

if __name__ == '__main__':
    renv = _prepare_run_env()
    _update_sys_paths(renv)
    if RELOAD_ALL_MODULES:
        _reload_all_modules(renv)

    debug_main(renv)
