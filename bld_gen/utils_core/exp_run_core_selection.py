import os
import importlib

def _find_unified_name(blender_prj_pathname):
    basename = os.path.basename(blender_prj_pathname)
    basename = basename.replace(".blend", "")
    unified_name = basename
    return unified_name

def _find_sub_folder(root_dir, unified_name):
    for leading in ["exp_", "bldmc_"]:
        rel_sub_exp_folder_name = "{}{}".format(leading, unified_name)

        sub_folder = "{}{}{}{}{}".format(root_dir, os.sep, "bld_gen", os.sep, rel_sub_exp_folder_name)
        if os.path.isdir(sub_folder):
            return sub_folder, rel_sub_exp_folder_name
    return None, None

def _find_run_core_py(sub_folder, rel_sub_exp_folder_name):
    sub_folder_core_py = "{}{}run_core.py".format(sub_folder, os.sep)
    if os.path.isfile(sub_folder_core_py):
        mod_name = "{}.run_core".format(rel_sub_exp_folder_name)
        return sub_folder_core_py, mod_name
    return None, None   

def _find_and_load_module(renv, blender_prj_pathname):
    root_dir = renv.get_root_dir()
    print("\n*find associated core for {} at root_dir {}".format(blender_prj_pathname, root_dir))

    unified_name = _find_unified_name(blender_prj_pathname)
    print("*search for sub_exp folder for", unified_name)

    sub_folder, rel_sub_exp_folder_name = _find_sub_folder(root_dir, unified_name)
    if sub_folder is None:
        raise ValueError("ERROR: No {} match {}".format(sub_folder, blender_prj_pathname))
    
    sub_folder_core_py, mod_name = _find_run_core_py(sub_folder, rel_sub_exp_folder_name)
    if sub_folder_core_py is None:
        raise ValueError("ERROR: No {} match {} under {}".format(sub_folder_core_py, blender_prj_pathname, sub_folder))
    
    print("*try to dyna load module {}\n".format(mod_name))
    try:
        mod_run_core = importlib.import_module(mod_name)
    except Exception as ex:
        raise ex

    importlib.reload(mod_run_core)
    return mod_run_core, sub_folder

def _find_entry_func(mod_run_core, action):
    from bld_gen.utils_ui.colorstr import log_colorstr
    if action is None or len(action) == 0 or action == "show":
        best_name = "do_exp"
    else:
        best_name = "do_exp_{}".format(action)
    for name in dir(mod_run_core):
        if name.startswith("do_exp"):
            if name == best_name:
                return getattr(mod_run_core, name)
    msg = "ERROR: No do_exp_{} function exposed by module {}".format(action, mod_run_core)
    log_colorstr("red", msg)
    log_colorstr("red", "ERROR: Please make sure you have selected right SELECTED_ACTION and also the run_core.py module has related do_exp_xxx exposed")
    raise ValueError(msg)

def _dyna_load_and_run(renv, action, blender_prj_pathname):
    from bld_gen.utils_ui.colorstr import log_colorstr
    import gc
    gc.collect(generation=2)

    mod_run_core, sub_run_folder = _find_and_load_module(renv, blender_prj_pathname)
    if mod_run_core is None:
        return False
    
    func_do_exp = _find_entry_func(mod_run_core, action)
    if func_do_exp is None:
        raise ValueError("{} is None".format(func_do_exp))

    if not callable(func_do_exp):
        raise ValueError("{} is not callable".format(func_do_exp))

    gc.collect(generation=2)
    log_colorstr("yellow", "### Run {} ...".format(func_do_exp))
    ret = func_do_exp(renv)
    log_colorstr("yellow", "### End {} ret: {}".format(func_do_exp, ret))
    print()
    return ret

def select_core_to_run(renv, action=""):
    import bpy

    # find run_core_module and supported actions
    filepath = bpy.data.filepath
    if os.path.isfile(filepath) and filepath.endswith(".blend"):
        ret = _dyna_load_and_run(renv, action, filepath)
    else:
        raise ValueError("blend file need to be open in blender before run the script")
    
    return ret
