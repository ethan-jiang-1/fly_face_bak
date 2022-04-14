import os
import sys
import importlib 

SELECTED_ACTION = "auto_render"  # show caputure auto_render
ENV_NEW = {"AUTO_RENDER":"auto_render.json",
           "ENV_AR_RENDER_ENGINE": "EEVEE",
           "ENV_AR_MAX_VARIATION": "-1",
           "ENV_AR_COMBINATION_ID": "-1"}
RELOAD_ALL_MODULES = True


class _RunEnv(object):
    @classmethod
    def init(cls):
        pass

    @classmethod
    def get_interface_dev_msg(cls):
        return None

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
            #raise ValueError("blend need to be open first")
            root_dir = os.path.dirname(host_dir)
            print("root_dir", root_dir)
            return root_dir

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

    bld_gen_dir = root_dir + os.sep + "bld_gen"
    if bld_gen_dir not in sys.path:
        sys.path.append(bld_gen_dir)

    print("updated sys.path")
    for path in sys.path:
        print(path)

def _prepare_run_env():
    import os 
    print("\n** Update Env")
    for key, val in ENV_NEW.items():
        if key not in os.environ:
            os.environ[key] = val
            print(" update {} = {}".format(key, val))
        else:
            print(" --skip {} = {} but keep {}".format(key, val, os.environ[key]))
    print()

    _RunEnv.init()
    return _RunEnv

def _dyn_load_to_run(renv, action):
    import bld_gen.utils_core.exp_run_core_selection as exp_run_core_selection
    print("reload main from main_debug")
    importlib.reload(exp_run_core_selection)
    exp_run_core_selection.select_core_to_run(renv, action=action)

def _reload_all_modules(renv):
    import bld_gen.utils_core.reload_all_modules as reload_all_modules
    print("reload reload_all_modules", reload_all_modules)
    importlib.reload(reload_all_modules)
    root_dir = renv.get_root_dir()
    reload_all_modules.reload_all_modules(root_dir)

def debug_main(renv=None, action=SELECTED_ACTION):
    if renv is None:
        renv = _prepare_run_env()

    _dyn_load_to_run(renv, action)

if __name__ == '__main__':
    renv = _prepare_run_env()
    _update_sys_paths(renv)
    if RELOAD_ALL_MODULES:
        _reload_all_modules(renv)

    debug_main(renv)
