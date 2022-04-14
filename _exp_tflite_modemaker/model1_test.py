# -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import time
from datetime import datetime

import tflite_runtime.interpreter as tflite

if __name__ == "__main__":
    dir_this = os.path.dirname(__file__)
    test_image_dir = '../_dataset_hair_styles/00'
    model_path = "{}/cls_hair_1.tflite".format(dir_this)
    dir_npy = "{}/_output".format(dir_this)
    os.makedirs(dir_npy, exist_ok=True)

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(str(input_details))
    output_details = interpreter.get_output_details()
    print(str(output_details))

    if 1:
        file_list = os.listdir(test_image_dir)

        model_interpreter_time = 0

        sum_np = []
        # 遍历文件
        for file in file_list:
            start_time = time.time()
            print('=========================')
            full_path = os.path.join(test_image_dir, file)
            print('full_path:{}'.format(full_path))

            #
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            # 增加一个维度，
            image_np_expanded = np.expand_dims(res_img, axis=0)
            # int to float32
            image_np_expanded = image_np_expanded.astype('float32')  # 类型也要满足要求

            # 填装数据
            model_interpreter_start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], image_np_expanded)

            # 调用模型
            d0 = datetime.now()
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            dt = datetime.now() - d0
            print("interence time: {:.3}sec".format(dt.total_seconds()))

            # 出来的结果去掉没用的维度
            output = np.squeeze(output_data)

            # get top1
            ordered = np.argsort(output)
            print(ordered[-1])
            for i in range(7):
                str_info = "%s %.2f%%" % ([ordered[i]], output[ordered[i]] * 100)
                print(str_info)

            used_time = time.time() - start_time
            print('used_time:{}'.format(used_time))

