from datetime import datetime 
import os

def _create_interpeter(model_pathname):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter
    return interpreter(model_path=model_pathname)

def _get_model_pathname():
    sub_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(sub_dir)
    paths = [root_dir, "ref_graph_unknown", "hair_segmentation.tflite"]
    #paths = [root_dir, "ref_graph_unknown", "hair_segmentation_junhwanjang.tflite"]
    full_path = os.sep.join(paths)
    if not os.path.isfile(full_path):
        raise ValueError("{} not exist".format(full_path))
    return full_path

def _chk_hair_segmentation():
    from utils_inspect.inspect_tflite import inspect_tflite
    model_pathname = _get_model_pathname()
    interpreter = _create_interpeter(model_pathname)
    print(interpreter)

    interpreter.allocate_tensors()
    inspect_tflite(interpreter)


def _add_parent_in_sys_path():
    import sys 
    import os 

    dir_this = os.path.dirname(__file__)
    dir_parent = os.path.dirname(dir_this)
    if dir_parent not in sys.path:
        sys.path.append(dir_parent)

def do_chk():
    d0 = datetime.now()
    print("loading mediapipe...")
    import mediapipe as mp
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    print(mp)
    _chk_hair_segmentation()


if __name__ == '__main__':
    _add_parent_in_sys_path()
    do_chk()
