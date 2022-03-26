# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:09:30 2022

@author: Administrator
"""

import os
import cv2
import math
import dlib
import numpy as np
import time

class DlibFaceAligment():
    predictor = None 

    @classmethod
    def get_predictor(cls):
        if cls.predictor is None:
            dir_this = os.path.dirname(__file__)
            dat_path = os.sep.join([dir_this, "shape_predictor_68_face_landmarks.dat"])
            if os.path.isfile(dat_path):
                cls.predictor = dlib.shape_predictor(dat_path) # 用来预测关键点
            else:
                raise ValueError("{} not exist".format(dat_path))
        return cls.predictor

    # def face_alignment(face,detection,w,h):
        
    #     left_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
    #     right_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

    #     eye_center =((left_eye.x + right_eye.x) * 1./2 * w, # 计算两眼的中心坐标
    #                   (left_eye.y + right_eye.y) * 1./2 * h)
    #     dx = (right_eye.x - left_eye.x) * width # note: right - right
    #     dy = (right_eye.y - left_eye.y) * height

    #     angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
    #     RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
    #     RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        
    #     return  RotImg

    @classmethod
    def make_face_alignment(cls, face, predictor=None, draw_keypoints=True):
        if predictor is None:
            predictor = cls.get_predictor()
        
        if draw_keypoints:
            rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
            shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
            order = [36, 45, 30, 48, 54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
                cv2.circle(face, (x, y), 2, (0, 255, 255), -1)

        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                    (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        return RotImg

    @classmethod
    def rect_to_bb(cls, detection, width, height): # 获得人脸矩形的坐标信息
        x = int(detection.location_data.relative_bounding_box.xmin * width)
        y = int(detection.location_data.relative_bounding_box.ymin * height)
        w = int(detection.location_data.relative_bounding_box.width * width)
        h = int(detection.location_data.relative_bounding_box.height * height)
        return (x, y, w, h)


def exam_face_aligment():
    from utils_inspect.sample_images import get_sample_images
    img_pathnames = get_sample_images()

    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    def get_output_filename(filename):
        basename = os.path.basename(filename)
        this_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(this_dir)
        output_dir = os.sep.join([root_dir, "_reserved_fa_output"])
        os.makedirs(output_dir, exist_ok= True)
        return os.sep.join([output_dir, basename])

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, filename in enumerate(img_pathnames):
            t0 = time.time()
            image = cv2.imread(filename)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print('wasted time %s' %(time.time() - t0))
            # Draw face detections of each face.
            if not results.detections:
                continue

            annotated_image = image.copy()
            height,width = annotated_image.shape[0],annotated_image.shape[1]
            
            for detection in results.detections:
                print('Nose tip:', mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))

                (x, y, w, h) = DlibFaceAligment.rect_to_bb(detection, width, height)
                detect_face = annotated_image[y:y+h,x:x+w]
                mp_drawing.draw_detection(annotated_image, detection)

                face_photo = DlibFaceAligment.make_face_alignment(detect_face)
                #face_photo = DlibFaceAligment.face_alignment(image)
            
            print(face_photo.shape, filename)
            cv2.imwrite(get_output_filename(filename), face_photo)

def _add_root_in_sys_path():
    import sys 
    import os
    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    if dir_root not in sys.path:
        sys.path.append(dir_root)

if __name__ == '__main__':
    _add_root_in_sys_path()
    exam_face_aligment()

