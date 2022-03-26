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

    # @classmethod
    # def face_alignment_mp(cls, face, detection, width, height):
    #     import mediapipe as mp
    #     mp_face_detection = mp.solutions.face_detection

    #     left_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
    #     right_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

    #     eye_center =((left_eye.x + right_eye.x) * 1./2 * width, # 计算两眼的中心坐标
    #                   (left_eye.y + right_eye.y) * 1./2 * height)
    #     dx = (right_eye.x - left_eye.x) * width # note: right - right
    #     dy = (right_eye.y - left_eye.y) * height

    #     angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
    #     RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
    #     RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        
    #     return  RotImg

    @classmethod
    def get_rotate_scalars_by_faceshape(cls, shape):
        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                    (shape.part(36).y + shape.part(45).y) * 1./2)
        
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)
        angle = math.atan2(dy, dx) * 180. / math.pi # 计算角度
        
        return eye_center, angle

    @classmethod
    def get_rotate_matrix_by_faceshape(cls, shape):
        eye_center, angle = cls.get_rotate_scalars_by_faceshape(shape)
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        return RotateMatrix

    @classmethod
    def make_face_alignment(cls, img_detected_face, predictor=None, draw_keypoints=True):
        if predictor is None:
            predictor = cls.get_predictor()

        rec = dlib.rectangle(0,0,img_detected_face.shape[0],img_detected_face.shape[1])
        shape = predictor(np.uint8(img_detected_face), rec) # 注意输入的必须是uint8类型  

        if draw_keypoints:
            order = [36, 45, 30, 48, 54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
                cv2.circle(img_detected_face, (x, y), 2, (0, 255, 255), -1)
            
            order = [19, 24, 16, 15, 7, 8, 9, 0, 1] # outline
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
                cv2.circle(img_detected_face, (x, y), 2, (255, 0, 255), -1)

        RotateMatrix = cls.get_rotate_matrix_by_faceshape(shape)

        RotImg = cv2.warpAffine(img_detected_face, RotateMatrix, (img_detected_face.shape[0], img_detected_face.shape[1])) # 进行放射变换，即旋转
        return RotImg

    @classmethod
    def make_picture_alignment(cls, image, predictor=None, draw_keypoints=False):
        if predictor is None:
            predictor = cls.get_predictor()

        imags_resized = cv2.resize(image, (512, 512))

        rec = dlib.rectangle(0,0,imags_resized.shape[0],imags_resized.shape[1])
        shape = predictor(np.uint8(imags_resized),rec) # 注意输入的必须是uint8类型  

        RotateMatrix = cls.get_rotate_matrix_by_faceshape(shape)

        RotImg = cv2.warpAffine(imags_resized, RotateMatrix, (512, 512)) # 进行放射变换，即旋转
        return RotImg        

class MpFaceDetection():
    @classmethod
    def rect_to_bb(cls, detection, width, height): # 获得人脸矩形的坐标信息
        x = int(detection.location_data.relative_bounding_box.xmin * width)
        y = int(detection.location_data.relative_bounding_box.ymin * height)
        w = int(detection.location_data.relative_bounding_box.width * width)
        h = int(detection.location_data.relative_bounding_box.height * height)
        return (x, y, w, h)

    @classmethod
    def rect_to_bb_expand(cls, detection, width, height, expand_ratio=0.1): # 获得人脸矩形的坐标信息
        x = int(detection.location_data.relative_bounding_box.xmin * width)
        y = int(detection.location_data.relative_bounding_box.ymin * height)
        w = int(detection.location_data.relative_bounding_box.width * width)
        h = int(detection.location_data.relative_bounding_box.height * height)

        x = int(x - width * expand_ratio) 
        if x < 0:
            x = 0

        y = int(y - height * expand_ratio)
        if y < 0:
            y = 0

        w = int(w + width * expand_ratio)
        if x + w > width:
            w = width - x 

        h = int(h + height * expand_ratio)
        if y + h > height:
            h = height - y

        return (x, y, w, h)

def exam_face_aligment(face_crop=True, draw_keypoints=True):
    from utils_inspect.sample_images import get_sample_images
    img_pathnames = get_sample_images()

    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    def get_output_filename(filename, face_crop):
        basename = os.path.basename(filename)
        this_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(this_dir)
        if face_crop:
            output_dir = os.sep.join([root_dir, "_reserved_fa_output"])
        else:
            output_dir = os.sep.join([root_dir, "_reserved_pa_output"])
        os.makedirs(output_dir, exist_ok= True)
        return os.sep.join([output_dir, basename])

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, filename in enumerate(img_pathnames):
            t0 = time.time()
            image = cv2.imread(filename)

            if face_crop:            
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                print('wasted time %s' %(time.time() - t0))
                # Draw face detections of each face.
                if not results.detections:
                    continue

                annotated_image = image.copy()
                height,width = annotated_image.shape[0],annotated_image.shape[1]
                
                #for detection in results.detections:
                mp_face_detection_result = results.detections[0]
                print('Nose tip:', mp_face_detection.get_key_point(mp_face_detection_result, mp_face_detection.FaceKeyPoint.NOSE_TIP))

                detected_face_bb = MpFaceDetection.rect_to_bb(mp_face_detection_result, width, height)
                #detected_face_bb = DlibFaceAligment.rect_to_bb_expand(detection, width, height)
                (x, y, w, h) = detected_face_bb
                img_detect_face = annotated_image[y:y+h,x:x+w]
                if draw_keypoints:
                    mp_drawing.draw_detection(annotated_image, mp_face_detection_result)

                aligned_pic = DlibFaceAligment.make_face_alignment(img_detect_face, draw_keypoints=draw_keypoints)
            else:
                aligned_pic = DlibFaceAligment.make_picture_alignment(image, draw_keypoints=draw_keypoints)
                #face_photo = DlibFaceAligment.face_alignment(image)
            
            print(aligned_pic.shape, filename)
            cv2.imwrite(get_output_filename(filename, face_crop), aligned_pic)


def _add_root_in_sys_path():
    import sys 
    import os
    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    if dir_root not in sys.path:
        sys.path.append(dir_root)

if __name__ == '__main__':
    _add_root_in_sys_path()
    #exam_face_aligment(face_crop=True)
    exam_face_aligment(face_crop=False)

