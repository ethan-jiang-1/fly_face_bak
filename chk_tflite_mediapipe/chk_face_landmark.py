from datetime import datetime 
import os

def _create_interpeter(model_pathname):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter
    return interpreter(model_path=model_pathname)

def _get_model_pathname():
    sub_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(sub_dir)
    paths = [root_dir, "ref_graph_exist", "face_landmark.tflite"]
    full_path = os.sep.join(paths)
    if not os.path.isfile(full_path):
        raise ValueError("{} not exist".format(full_path))
    return full_path

def _chk_face_landmark():
    model_pathname = _get_model_pathname()
    interpreter = _create_interpeter(model_pathname)
    print(interpreter)

    interpreter.allocate_tensors()

    print("== Input details ==")
    print("name:", interpreter.get_input_details()[0]['name'])
    print("shape:", interpreter.get_input_details()[0]['shape'])
    print("type:", interpreter.get_input_details()[0]['dtype'])

    print("\nDUMP INPUT")
    print(interpreter.get_input_details()[0])

    print("\n== Output details ==")
    print("name:", interpreter.get_output_details()[0]['name'])
    print("shape:", interpreter.get_output_details()[0]['shape'])
    print("type:", interpreter.get_output_details()[0]['dtype'])

    print("\nDUMP OUTPUT")
    print(interpreter.get_output_details()[0])

def _inspect_mp(mp):
    from utils_inspect.inspect_mp import inspect_mp
    inspect_mp(mp)

def _inspect_cv2(cv2):
    print(cv2)

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

    _inspect_mp(mp)
    _inspect_cv2(cv2)
    _chk_face_landmark()


def _add_parent_in_sys_path():
    import sys 
    import os 

    dir_this = os.path.dirname(__file__)
    dir_parent = os.path.dirname(dir_this)
    if dir_parent not in sys.path:
        sys.path.append(dir_parent)

if __name__ == '__main__':
    _add_parent_in_sys_path()
    do_chk()
