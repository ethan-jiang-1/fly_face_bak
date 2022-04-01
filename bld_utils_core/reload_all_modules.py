import os
import sys # we use sys.path
import importlib # we use importlib.import_module

MODLES_EXCLUED = ["bld_utils_core.reload_all_modules",
                  "bld_utils_model.easy_dict"]

def _reload_all_py_in_subfolder(root_dir, subfolder):
    import_folder = "{}{}{}".format(root_dir, os.sep, subfolder)
    src_files = os.listdir(import_folder)

    module_names = list(sys.modules.keys())

    #indentify how many modules require reloaded
    reload_mod_names = []
    for src_file in src_files:
        if not src_file.endswith(".py"):
            continue
  
        src_stem = src_file[:-3]
        abs_name = import_folder + os.sep + src_stem
        rel_name = abs_name[len(root_dir) + 1:]
        mod_name = rel_name.replace(os.sep, ".")

        if mod_name in MODLES_EXCLUED:
            continue

        if mod_name in module_names:
            reload_mod_names.append(mod_name)
    
    # reload the same order as previous loaded
    for mod_name in module_names:
        if mod_name in reload_mod_names:
            importlib.import_module(mod_name)
            importlib.reload(sys.modules[mod_name])
            importlib.import_module(mod_name)

def _find_sub_folders(root_dir):
    reserved_folders = ["bld_model_inspect", "bld_utils_core", "bld_utils_model", "bld_utils_ui"]
    names = os.listdir(root_dir)

    exp_folders = []
    for name in names:
        dir_name = root_dir + os.sep + name 
        if not os.path.isdir(dir_name):
            continue

        if name.startswith("exp_"):
            exp_folders.append(name)

    return reserved_folders + exp_folders

#the purpose of reload_all_modules is to refresh the module in vc in debug mode
def reload_all_modules(root_dir):
    if root_dir is None or len(root_dir) < 3:
        return 

    if root_dir not in sys.path:
        sys.path.append(root_dir)

    print("\n##reload modules")
    subfolders = _find_sub_folders(root_dir)
    for subfolder in subfolders:
        _reload_all_py_in_subfolder(root_dir, subfolder)
    print("\n")
