import os
import sys 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
from datetime import datetime

import tflite_runtime.interpreter as tflite

# Load TFLite model and allocate tensors.
dir_this = os.path.dirname(__file__)
model_path_M = "{}/cls_hair_M.tflite".format(dir_this)
model_path_F = "{}/cls_hair_F.tflite".format(dir_this)
interpreter_M = tflite.Interpreter(model_path=model_path_M)
interpreter_F = tflite.Interpreter(model_path=model_path_F)
interpreter_M.allocate_tensors()
interpreter_F.allocate_tensors()
# Get input and output tensors.
input_details_M = interpreter_M.get_input_details()
input_details_F = interpreter_F.get_input_details()
# print(str(input_details))
output_details_M = interpreter_M.get_output_details()
output_details_F = interpreter_F.get_output_details()
# print(str(output_details))
M_map = {'0':0,'1':2,'2':3,'3':6}
F_map = {'0':1,'1':4,'2':5}

class ClfHair(object):
    @classmethod
    def get_category_id(cls, img_hair, gender="M"):
        if gender == "M":
            img = img_hair
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # reshape
            res_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            interpreter_M.reset_all_variables()

            # expand 1 axis
            image_np_expanded = np.expand_dims(res_img, axis=0)
            image_np_expanded = image_np_expanded.astype('float32')
            # feed data
            interpreter_M.set_tensor(input_details_M[0]['index'], image_np_expanded)
            # run model
            d0 = datetime.now()
            interpreter_M.invoke()
            output_data = interpreter_M.get_tensor(output_details_M[0]['index'])
            dt = datetime.now() - d0
            print("interence time(clf_hair): {:.3}sec".format(dt.total_seconds()))
            #
            output = np.squeeze(output_data)
            # get top1
            ordered = np.argsort(output)
            print("the classid is %s" %ordered[-1])
            # M classfiy
            order_id = M_map[str(ordered[-1])]
        else:
            img = img_hair
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # reshape
            res_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            interpreter_F.reset_all_variables()
    
            # expand 1 axis
            image_np_expanded = np.expand_dims(res_img, axis=0)
            image_np_expanded = image_np_expanded.astype('float32')
            # feed data
            interpreter_F.set_tensor(input_details_F[0]['index'], image_np_expanded)
            # run model
            d0 = datetime.now()
            interpreter_F.invoke()
            output_data = interpreter_F.get_tensor(output_details_F[0]['index'])
            dt = datetime.now() - d0
            print("interence time(clf_hair): {:.3}sec".format(dt.total_seconds()))
            #
            output = np.squeeze(output_data)
            # get top1
            ordered = np.argsort(output)
            print("the classid is %s" % ordered[-1])
            # M classfiy
            order_id = F_map[str(ordered[-1])]
        return order_id

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
