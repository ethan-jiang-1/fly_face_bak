import os
import sys 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
from datetime import datetime

import tflite_runtime.interpreter as tflite

# Load TFLite model and allocate tensors.
dir_this = os.path.dirname(__file__)
model_path = "{}/cls_hair_1.tflite".format(dir_this)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
# print(str(input_details))
output_details = interpreter.get_output_details()
# print(str(output_details))

class ClfHair(object):
    @classmethod
    def get_category_id(cls, img_bin_hair):
        img = img_bin_hair
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # reshape
        res_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # expand 1 axis
        image_np_expanded = np.expand_dims(res_img, axis=0)
        image_np_expanded = image_np_expanded.astype('float32')
        # feed data
        interpreter.set_tensor(input_details[0]['index'], image_np_expanded)
        # run model
        d0 = datetime.now()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        dt = datetime.now() - d0
        print("interence time: {:.3}sec".format(dt.total_seconds()))
        #
        output = np.squeeze(output_data)
        # get top1
        ordered = np.argsort(output)
        print("the classid is %s" %ordered[-1])

        return ordered[-1]

def do_exp(img_pathname):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not img_pathname.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        img_pathname = "{}/{}".format(dir_root, img_pathname)

    img_array = cv2.imread(img_pathname)
    id = ClfHair.get_category_id(img_array)
    print("category id for {} is {}".format(img_pathname, id))
    return id

if __name__ == '__main__':
    filename = "_reserved_output_feature_gen/06_000_0000_unified_shift_full.jpg"
    do_exp(filename)
