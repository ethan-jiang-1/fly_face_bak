import os
import cv2
import numpy as np
#import pandas as pd
from face_aligment_base import FaceAligmentBase


class FaceAlignemtMp(FaceAligmentBase):
    def __init__(self, debug=True):
        super(FaceAlignemtMp, self).__init__(debug=debug)

    def create_detector(self):
        import mediapipe as mp # this is not a must dependency. do not import it in the global level.
        mp_face_mesh = mp.solutions.face_mesh
        self.slt_face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.face_oval_ndx = []
        for ct in mp.solutions.face_mesh_connections.FACEMESH_FACE_OVAL:
            fpt, tpt = ct 
            if fpt not in self.face_oval_ndx:
                self.face_oval_ndx.append(fpt)

    def close_detector(self):
        self.slt_face_mesh.close()
        del self.slt_face_mesh

    def _get_key_landmark(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.slt_face_mesh.reset()
        results = self.slt_face_mesh.process(img)
        if results is None:
            return None
        if results.multi_face_landmarks is None:
            return None
        if len(results.multi_face_landmarks) == 0:
            return None
        face_landmark = results.multi_face_landmarks[0]
        return face_landmark

    def _get_landmark_bounding(self, llm):
        landmark_face_oval_x = []
        landmark_face_oval_y = []
        for ndx in self.face_oval_ndx:
            llmn = llm[ndx]
            if (llmn.x >= 0) and (llmn.y >= 0) and (llmn.x <= 1) and (llmn.y <= 1):
                landmark_face_oval_x.append(llmn.x)
                landmark_face_oval_y.append(llmn.y)
            else:
                print("invalid pt at ", ndx)
        np_landmark_face_oval_x = np.array(landmark_face_oval_x)
        np_landmark_face_oval_y = np.array(landmark_face_oval_y)

        x_max = np_landmark_face_oval_x.max()
        x_min = np_landmark_face_oval_x.min()
        y_max = np_landmark_face_oval_x.max()
        y_min = np_landmark_face_oval_y.min()
        return x_min, y_min, x_max-x_min, y_max-y_min
        
    def find_and_crop_face(self, img, draw_marks=False):
        landmark = self._get_key_landmark(img)
        if landmark is None:
            return None

        height,width = img.shape[0],img.shape[1]

        b_xmin, b_ymin, b_height, b_width = self._get_landmark_bounding(landmark.landmark)
        x = int(b_xmin * width)
        y = int(b_ymin * height)
        w = int(b_width * width)
        h = int(b_height * height)
        img_crop = img[y:y+h,x:x+w]
        return img_crop

    def detect_face_and_eyes(self, img_org):
        landmark = self._get_key_landmark(img_org)
        if landmark is None:
            return None, None, None
    
        height,width = img_org.shape[0],img_org.shape[1]

        b_xmin, b_ymin, b_height, b_width = self._get_landmark_bounding(landmark.landmark)
        x = int(b_xmin * width)
        y = int(b_ymin * height)
        w = int(b_width * width)
        h = int(b_height * height)

        #le = mp.solutions.face_mesh_connections.FACEMESH_LEFT_EYE   # 362<->263
        #re = mp.solutions.face_mesh_connections.FACEMESH_RIGHT_EYE  # 33<->133
        
        llm = landmark.landmark

        abs_right_eye = (int((llm[33].x * width + llm[133].x * width) / 2), int((llm[33].y * height + llm[133].y * height)/2))
        abs_left_eye = (int((llm[362].x * width + llm[263].x * width) / 2), int((llm[362].y * height + llm[263].y * height)/2))
        right_eye = (abs_right_eye[0]-x, abs_right_eye[1]-y)
        left_eye = (abs_left_eye[0]-x, abs_left_eye[1]-y)

        img_crop = img_org[y:y+h,x:x+w]
        #return img_org[y:y+h,x:x+w], left_eye, right_eye
        return img_crop, right_eye, left_eye
    

def exam_face_aligment(face_crop=False, selected_names=None):
    from utils_inspect.sample_images import get_sample_images
    test_set = get_sample_images()

    fa = FaceAlignemtMp()
    fa.create_detector()

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
    fa.close_detector()

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
