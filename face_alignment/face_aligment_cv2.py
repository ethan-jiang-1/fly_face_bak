import os
import cv2
import numpy as np
import pandas as pd
from face_aligment_base import FaceAligmentBase


class FaceAlignemtCv2(FaceAligmentBase):
    def __init__(self, debug=True):
        super(FaceAlignemtCv2, self).__init__(debug=debug)

        self._create_cv_detectors()

    #opencv path
    def _create_cv_detectors(self):
        opencv_home = cv2.__file__
        folders = opencv_home.split(os.path.sep)[0:-1]

        path = folders[0]
        for folder in folders[1:]:
            path = path + "/" + folder

        face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
        eye_detector_path = path+"/data/haarcascade_eye.xml"
        nose_detector_path = path+"/data/haarcascade_mcs_nose.xml"

        if not os.path.isfile(face_detector_path):
            raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")

        self.face_detector = cv2.CascadeClassifier(face_detector_path)
        self.eye_detector = cv2.CascadeClassifier(eye_detector_path) 
        self.nose_detector = cv2.CascadeClassifier(nose_detector_path) 

    def detect_face(self, img):
        for att in [0, 1, 2]:
            if att == 0:
                faces = self.face_detector.detectMultiScale(img, 1.1, 10)
            elif att == 1:
                faces = self.face_detector.detectMultiScale(img, 1.3, 5)
            elif att == 2:
                faces = self.face_detector.detectMultiScale(img)
            if len(faces) >= 1:
                break

        if len(faces) > 0:
            face = faces[0]
            face_x, face_y, face_w, face_h = face
            img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
            return img
        
        return None

    def detect_face_and_eys(self, img_org):
        img_face = self.detect_face(img_org)
        if img_face is None:
            return None, None 

        img_gray_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
        
        for att in [0, 1, 2]:
            if att == 0:
                eyes = self.eye_detector.detectMultiScale(img_gray_face, 1.1, 10)
            elif att == 1:
                eyes = self.eye_detector.detectMultiScale(img_gray_face, 1.3, 5)
            elif att == 2:
                eyes = self.eye_detector.detectMultiScale(img_gray_face)

            eyes = sorted(eyes, key = lambda v: abs((v[0] - v[2]) * (v[1] - v[3])), reverse=True)
            eyes =np.array(eyes)
            if len(eyes) >= 2:
                break
        
        if len(eyes) >= 2:
            #find the largest 2 eye
            
            base_eyes = eyes[:, 2]
            #print(base_eyes)
            
            items = []
            for i in range(0, len(base_eyes)):
                item = (base_eyes[i], i)
                items.append(item)
            
            df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
            
            eyes = eyes[df.idx.values[0:2]]
        else:
            eyes = None

        return img_face, eyes
    

def exam_face_aligment(face_crop=False, selected_names=None):
    from utils_inspect.sample_images import get_sample_images
    test_set = get_sample_images()

    fa = FaceAlignemtCv2()

    for filename in test_set:
        basename = os.path.basename(filename)
        if selected_names is not None:
            if basename not in selected_names:
                continue
        
        img_org = cv2.imread(filename)
        fa_ret = fa.align_face(img_org)
        if fa_ret is None:
            continue

        fa.plot_fa_result(fa_ret, basename)

def _add_root_in_sys_path():
    import sys 
    import os
    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    if dir_root not in sys.path:
        sys.path.append(dir_root)

if __name__ == '__main__':
    _add_root_in_sys_path()

    selected_names = None 
    #selected_names = ["hsi_image4.jpeg"]
    #selected_names = ["hsi_image1.jpeg"]
    exam_face_aligment(face_crop=True, selected_names=selected_names)
    #exam_face_aligment(face_crop=False)
