from datetime import datetime 
import os

def _create_interpeter(model_pathname):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter
    return interpreter(model_path=model_pathname)

def _get_model_pathname():
    sub_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(sub_dir)
    #paths = [root_dir, "ref_graph_unknown", "hair_segmentation.tflite"]
    paths = [root_dir, "ref_graph_unknown", "hair_segmentation_junhwanjang.tflite"]
    full_path = os.sep.join(paths)
    if not os.path.isfile(full_path):
        raise ValueError("{} not exist".format(full_path))
    return full_path

def _chk_hair_segmentation():
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
    _chk_hair_segmentation()


if __name__ == '__main__':
    do_chk()
