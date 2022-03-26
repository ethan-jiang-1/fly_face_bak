# -*- coding: utf-8 -*-

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
    def create_slt_face_detection(cls):
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        slt_face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        return slt_face_detection 

    @classmethod
    def close_slt_face_detection(cls, slt_face_detection):
        if slt_face_detection is not None:
            slt_face_detection.close()

    @classmethod
    def get_key_dectection_result(cls, mp_face_detection_results):
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        if mp_face_detection_results is None:
            return None
        if len(mp_face_detection_results.detections) == 0:
            return None

        key_face_detection_result = mp_face_detection_results.detections[0]
        print('Nose tip:', mp_face_detection.get_key_point(key_face_detection_result, mp_face_detection.FaceKeyPoint.NOSE_TIP))
        return key_face_detection_result

    @classmethod
    def _rect_to_bb(cls, detection, width, height): # 获得人脸矩形的坐标信息
        x = int(detection.location_data.relative_bounding_box.xmin * width)
        y = int(detection.location_data.relative_bounding_box.ymin * height)
        w = int(detection.location_data.relative_bounding_box.width * width)
        h = int(detection.location_data.relative_bounding_box.height * height)
        return (x, y, w, h)

    @classmethod
    def _rect_to_bb_expand(cls, detection, width, height, expand_ratio=0.1): # 获得人脸矩形的坐标信息
        (x, y, w, h) = cls._rect_to_bb(detection, width, height)
        nx = int(x - width * expand_ratio) 
        if nx < 0:
            nx = 0

        ny = int(y - height * expand_ratio)
        if ny < 0:
            ny = 0

        nw = int(w + width * expand_ratio)
        if nx + nw > width:
            nw = width - nx 

        nh = int(h + height * expand_ratio)
        if ny + nh > height:
            nh = height - ny

        return (nx, ny, nw, nh)

    @classmethod
    def draw_keypoints_detect_face(cls, annotated_image, key_face_detection_result, detected_face_bb, detected_face_bb_exp):
        import mediapipe as mp
        import mediapipe.python.solutions.drawing_utils as du
        import mediapipe.python.solutions.drawing_styles as ds 
        keypoint_drawing_spec = du.DrawingSpec(color=ds._RED, thickness=2, circle_radius=2)
        bbox_drawing_spec = du.DrawingSpec(color=ds._PEACH, thickness=0)
        mp.solutions.drawing_utils.draw_detection(annotated_image, key_face_detection_result, keypoint_drawing_spec, bbox_drawing_spec)

    @classmethod
    def get_cropped_img_face(cls, image, key_face_detection_result, draw_keypoints=False):
        annotated_image = image.copy()
        height,width = annotated_image.shape[0],annotated_image.shape[1]

        detected_face_bb = cls._rect_to_bb(key_face_detection_result, width, height)
        detected_face_bb_exp = cls._rect_to_bb_expand(key_face_detection_result, width, height, expand_ratio=0.0)
        (nx, ny, nw, nh) = detected_face_bb_exp
        print(detected_face_bb, detected_face_bb_exp)
        img_detect_face = annotated_image[ny:ny+nh,nx:nx+nw]
        #if draw_keypoints:
        #    cls.draw_keypoints_detect_face(annotated_image, key_face_detection_result, detected_face_bb, detected_face_bb_exp)

        return img_detect_face

def exam_face_aligment(face_crop=True, draw_keypoints=True):
    from utils_inspect.sample_images import get_sample_images
    img_pathnames = get_sample_images()

    def get_output_filename(filename, face_crop):
        basename = os.path.basename(filename)
        this_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(this_dir)
        if face_crop:
            output_dir = os.sep.join([root_dir, "_reserved_output_face_aligned"])
        else:
            output_dir = os.sep.join([root_dir, "_reserved_output_pic_aligned"])
        os.makedirs(output_dir, exist_ok= True)
        return os.sep.join([output_dir, basename])

    slt_face_detection = None 
    for idx, filename in enumerate(img_pathnames):
        print("process {} {} face_crop: {}".format(idx, filename, face_crop))
        t0 = time.time()
        image = cv2.imread(filename)

        if face_crop:
            if slt_face_detection is None: 
                slt_face_detection = MpFaceDetection.create_slt_face_detection()    

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = slt_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print('wasted time %s' %(time.time() - t0))

            if not results.detections:
                print("no face detected")
                continue

            key_face_detection_result = MpFaceDetection.get_key_dectection_result(results)    
            if key_face_detection_result is None:
                print("no key face detected")
                continue

            img_detect_face = MpFaceDetection.get_cropped_img_face(image, key_face_detection_result, draw_keypoints=draw_keypoints)

            aligned_pic = DlibFaceAligment.make_face_alignment(img_detect_face, draw_keypoints=draw_keypoints)
        else:
            aligned_pic = DlibFaceAligment.make_picture_alignment(image, draw_keypoints=draw_keypoints)
        
        output_pathname = get_output_filename(filename, face_crop)
        cv2.imwrite(output_pathname, aligned_pic)
        print("{} saved {}".format(output_pathname, aligned_pic.shape))
    
    MpFaceDetection.close_slt_face_detection(slt_face_detection)


def _add_root_in_sys_path():
    import sys 
    import os
    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    if dir_root not in sys.path:
        sys.path.append(dir_root)

if __name__ == '__main__':
    _add_root_in_sys_path()
    exam_face_aligment(face_crop=True)
    #exam_face_aligment(face_crop=False)

