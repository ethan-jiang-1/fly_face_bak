import importlib
import traceback

ACTION = "register"
#ACTION = "unregister"

#ADDON_DEV_MODULE = "dev_confirmation"
ADDON_DEV_MODULE = "dev_msg_panel"
#ADDON_DEV_MODULE = "dev_modal"


def _find_register_module():
    if ADDON_DEV_MODULE == "dev_confirmation":
        import bld_gen.utils_core.addon_dev_confirmation as addon_dev_confirmation
        importlib.reload(addon_dev_confirmation)
        return addon_dev_confirmation
    elif ADDON_DEV_MODULE == "dev_msg_panel":
        import bld_gen.utils_core.addon_dev_msg_panel as addon_dev_msg_panel
        importlib.reload(addon_dev_msg_panel)
        return addon_dev_msg_panel
    elif ADDON_DEV_MODULE == "dev_modal":
        import bld_gen.utils_core.addon_dev_modal as addon_dev_modal
        importlib.reload(addon_dev_modal)
        return addon_dev_modal
    return None

def dyn_load_and_register():
    mod_addon = _find_register_module()
    if mod_addon is None: 
        raise ValueError("module not found {}".format(ADDON_DEV_MODULE))

    register = mod_addon .register
    print("register func", register)

    try:
        register()
    except Exception as ex:
        print("Exception occured", ex)
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        raise ex

    if hasattr(mod_addon, "test_ui"):
        mod_addon.test_ui()

def dyn_load_and_unregister():
    mod_addon = _find_register_module()
    if mod_addon is None: 
        raise ValueError("module not found")

    unregister = mod_addon.unregister
    print("unregister func", unregister)

    try:
        unregister()
    except Exception as ex:
        print("Exception occured", ex)
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        raise ex

def exec_main():
    if ACTION == "register":
        dyn_load_and_register()
    elif ACTION == "unregister":
        dyn_load_and_unregister()
    

if __name__ == '__main__':
    exec_main()

