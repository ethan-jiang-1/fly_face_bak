from datetime import datetime 
import os

def _create_interpeter(model_pathname):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter
    return interpreter(model_path=model_pathname)

def _get_model_pathname():
    sub_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(sub_dir)
    paths = [root_dir, "ref_graph_exist", "selfie_segmentation.tflite"]
    full_path = os.sep.join(paths)
    if not os.path.isfile(full_path):
        raise ValueError("{} not exist".format(full_path))
    return full_path

def _chk_selfie_segmentation():
    from utils_inspect.inspect_tflite import inspect_tflite
    model_pathname = _get_model_pathname()
    interpreter = _create_interpeter(model_pathname)
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

    d0 = datetime.now()
    import cv2 as cv2
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    print(mp)
    print(cv2)
    _chk_selfie_segmentation()


if __name__ == '__main__':
    _add_parent_in_sys_path()
    do_chk()
