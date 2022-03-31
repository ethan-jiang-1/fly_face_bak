# -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import time

import tflite_runtime.interpreter as tflite

if __name__ == "__main__":
    test_image_dir = '../_reserved_output_feature_gen'
    model_path = "MobileFaceNet_epoch_4_iter_16000.tflite"

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

        # 遍历文件
        for file in file_list:
            start_time = time.time()
            print('=========================')
            full_path = os.path.join(test_image_dir, file)
            print('full_path:{}'.format(full_path))

            # 只要黑白的，大小控制在(28,28)
            img = cv2.imread(full_path)
            res_img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)

            # 增加一个维度，
            image_np_expanded = np.expand_dims(res_img, axis=0)
            # int to float32
            image_np_expanded = image_np_expanded.astype('float32')  # 类型也要满足要求

            # 填装数据
            model_interpreter_start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], image_np_expanded)

            # 调用模型
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            model_interpreter_time += time.time() - model_interpreter_start_time

            # 出来的结果去掉没用的维度
            result = np.squeeze(output_data)
            # save
            np.save('./npy/{}.npy'.format(file.split('.')[0]),result)

            # print('result:{}'.format(result))
            # print('result:{}'.format(sess.run(output, feed_dict={newInput_X: image_np_expanded})))
            used_time = time.time() - start_time
            print('used_time:{}'.format(used_time))
            print('model_interpreter_time:{}'.format(model_interpreter_time))


