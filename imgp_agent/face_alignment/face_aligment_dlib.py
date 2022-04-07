import os
import cv2
import numpy as np
import dlib
try:
    from .face_aligment_base import FaceAligmentBase
except:
    from face_aligment_base import FaceAligmentBase


class FaceAlignemtDlib(FaceAligmentBase):
    def __init__(self, debug=True):
        super(FaceAlignemtDlib, self).__init__(debug=debug)
        self.predictor = None

        self.slt_face_detection = MpFaceDetectionHelper.create_slt_face_detection() 

    def create_detector(self):
        if self.predictor is None:
            dir_this = os.path.dirname(__file__)
            dat_path = os.sep.join([dir_this, "shape_predictor_68_face_landmarks.dat"])
            if os.path.isfile(dat_path):
                self.predictor = dlib.shape_predictor(dat_path) # 用来预测关键点
            else:
                raise ValueError("{} not exist".format(dat_path))
        return self.predictor

    def close_detector(self):
        del self.predictor
        MpFaceDetectionHelper.close_slt_face_detection(self.slt_face_detection)

    def detect_face_and_eyes(self, img_org):

        self.slt_face_detection.reset()
        results = self.slt_face_detection.process(cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB))

        key_face_detection_result = MpFaceDetectionHelper.get_key_dectection_result(results) 

        img_face, bbox_face_in_org = MpFaceDetectionHelper.get_cropped_img_face(img_org, key_face_detection_result, draw_keypoints=False)

        right_eye,  left_eye = self._detect_eyes(img_face)

        right_eye_in_face = left_eye
        left_eye_in_face = right_eye

        return bbox_face_in_org, img_face, left_eye_in_face, right_eye_in_face 

    def _detect_eyes(self, img):
        rec = dlib.rectangle(0,0,img.shape[0],img.shape[1])
        shape = self.predictor(np.uint8(img),rec) # 注意输入的必须是uint8类型 

        right_eye = (int((shape.part(36).x + shape.part(39).x)/2), int((shape.part(36).y + shape.part(39).y)/2))
        left_eye = (int((shape.part(42).x + shape.part(45).x)/2), int((shape.part(42).y + shape.part(45).y)/2))    
        
        return right_eye,  left_eye

class MpFaceDetectionHelper():
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
    def get_cropped_img_face(cls, image, key_face_detection_result, draw_keypoints=False):
        annotated_image = image.copy()
        height,width = annotated_image.shape[0],annotated_image.shape[1]

        detected_face_bb = cls._rect_to_bb(key_face_detection_result, width, height)
        detected_face_bb_exp = cls._rect_to_bb_expand(key_face_detection_result, width, height, expand_ratio=0.0)
        (nx, ny, nw, nh) = detected_face_bb_exp
        print(detected_face_bb, detected_face_bb_exp)
        img_detect_face = annotated_image[ny:ny+nh,nx:nx+nw]

        return img_detect_face, (nx, ny, nw, nh)


def exam_face_aligment(face_crop=False, selected_names=None, show_summary=False):
    from utils_inspect.sample_images import get_sample_images
    test_set = get_sample_images()

    fa = FaceAlignemtDlib()
    fa.create_detector()

    sum =[]
    for filename in test_set:
        basename = os.path.basename(filename)
        if selected_names is not None:
            if basename not in selected_names:
                continue
        
        img_org = cv2.imread(filename)
        fa_ret = fa.align_face(img_org, img_name=basename)
        if fa_ret is None:
            continue
        if show_summary:
            fa.update_fa_sum(sum, fa_ret)
        else:
            fa.plot_fa_result(fa_ret, basename)
    if show_summary:
        fa.plot_fa_sum(sum)
    fa.close_detector()

def _add_root_in_sys_path():
    import sys 
    import os
    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(os.path.dirname(dir_this))
    if dir_root not in sys.path:
        sys.path.append(dir_root)

if __name__ == '__main__':
    _add_root_in_sys_path()

    selected_names = None 
    #selected_names = ["hsi_image4.jpeg"]
    #selected_names = ["hsi_image1.jpeg"]
    #selected_names = ["icl_image2.jpeg"]

    exam_face_aligment(face_crop=True, selected_names=selected_names)
    #exam_face_aligment(face_crop=False)
