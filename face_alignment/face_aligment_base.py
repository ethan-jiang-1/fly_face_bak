from collections import namedtuple
import math
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime 


FA_RESULT = namedtuple('FA_RESULT', 'img_org img_face img_rotated img_ratoted_unified dt')
FA_IMG_SIZE = 512
FA_TRANSINFO = namedtuple("FA_TRANSINFO", 'img_org_shape bbox_crop img_crop_shape direction angle center_of_eyes distance_of_eyes')

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

class FaceAligmentBase(object):
    def __init__(self, debug=False):
        self.debug = debug

    def create_detector():
        raise ValueError("not-implemented")

    def close_detector():
        raise ValueError("not-implemented")

    def detect_face_and_eyes(self, img_org):
        raise ValueError("not-implemented")

    def _compute_eyes_location(self, left_eye, right_eye):
        #center of eyes
        if len(left_eye) == 4:
            left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        else:
            left_eye_center = left_eye

        if len(right_eye) == 4:
            right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        else:
            right_eye_center = right_eye
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        center_of_eyes = (int((left_eye_x+right_eye_x)/2), int((left_eye_y+right_eye_y)/2))
        return left_eye_center, right_eye_center, center_of_eyes

    def _find_transform_info(self, img_org, bbox_crop_in_org, img_crop, left_eye_in_crop, right_eye_in_crop):
        left_eye_center, right_eye_center, center_of_eyes = self._compute_eyes_location(left_eye_in_crop, right_eye_in_crop) 
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        # find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 # rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 # rotate inverse direction of clock
            print("rotate to inverse clock direction")
        
        if self.debug:       
            cv2.circle(img_crop, left_eye_center, 2, (255, 0, 0) , 2)
            cv2.circle(img_crop, right_eye_center, 2, (255, 255, 0) , 2)
            cv2.circle(img_crop, center_of_eyes, 2, (255, 0, 255) , 2)

            cv2.circle(img_crop, point_3rd, 2, (255, 0, 0) , 2)
            
            cv2.line(img_crop,right_eye_center, left_eye_center,(67,67,67),1)
            cv2.line(img_crop,left_eye_center, point_3rd,(67,67,67),1)
            cv2.line(img_crop,right_eye_center, point_3rd,(67,67,67),1)

        da = euclidean_distance(left_eye_center, point_3rd)
        db = euclidean_distance(right_eye_center, point_3rd)
        dc = euclidean_distance(right_eye_center, left_eye_center)

        cos_a = (db*db + dc*dc - da*da)/(2*db*dc)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi
        if direction == -1:
            angle = 90 - angle
        
        #print("left eye: ", left_eye_center)
        #print("right eye: ", right_eye_center)
        #print("additional point: ", point_3rd)
        #print("triangle lengths: ",da, db, dc)
        #print("angle: ", angle," in degree")

        fti = FA_TRANSINFO(img_org_shape=img_org.shape,
                           bbox_crop=bbox_crop_in_org, 
                           img_crop_shape=img_crop.shape, 
                           direction=direction, 
                           angle=angle,
                           center_of_eyes=center_of_eyes,
                           distance_of_eyes=dc)
        return fti

    def align_face(self, img_org):
        d0 = datetime.now()
        img_raw = img_org.copy()
        
        bbox_crop_in_org, img_crop, left_eye_in_crop, right_eye_in_crop = self.detect_face_and_eyes(img_raw)

        if img_crop is None:
            return FA_RESULT(img_org=img_org, 
                              img_face=None, 
                              img_rotated=None, 
                              img_ratoted_unified=None,
                              dt=datetime.now() - d0) 

        if right_eye_in_crop is None and left_eye_in_crop is None:
            return FA_RESULT(img_org=img_org, 
                              img_face=img_crop, 
                              img_rotated=None, 
                              img_ratoted_unified=None,
                              dt=datetime.now() - d0) 
        
        #rotate image
        fti = self._find_transform_info(img_org, bbox_crop_in_org, img_crop, left_eye_in_crop, right_eye_in_crop)
        direction, angle = fti.direction, fti.angle
        
        img_rotated = Image.fromarray(img_raw)
        img_rotated = np.array(img_rotated.rotate(direction * angle))

        img_ratoted_unified = self.transfer_to_unified(img_raw, fti)

        dt = dt=datetime.now() - d0
        print("inference time: {:.3f}".format(dt.total_seconds()))
        return FA_RESULT(img_org=img_org, 
                         img_face=img_crop, 
                         img_rotated=img_rotated, 
                         img_ratoted_unified=img_ratoted_unified,
                         dt=dt)

    def transfer_to_unified(self, img_raw, fti):
        # img_face, _ = self.find_and_crop_face(img_raw)
        
        # img_rotated = Image.fromarray(img_face)
        # img_rotated = np.array(img_rotated.rotate(fti.direction * fti.angle))
        # return img_rotated
        return None

    def plot_fa_result(self, fa_ret, img_name):
        if fa_ret is None:
            return
        
        fig = plt.figure(figsize=(12, 6))
        plt.title(img_name)
        ax = plt.gca()
        ax.set_axis_off()

        ax = fig.add_subplot(1,4,1)
        ax.set_title("org")
        ax.set_axis_off()
        ax.imshow(fa_ret.img_org)

        if fa_ret.img_face is not None:
            ax = fig.add_subplot(1,4,2)
            ax.set_title("face")
            ax.set_axis_off()
            ax.imshow(fa_ret.img_face)

        if fa_ret.img_rotated is not None:
            ax = fig.add_subplot(1,4,3)
            ax.set_axis_off()
            ax.set_title("rotated")
            ax.imshow(fa_ret.img_rotated[:, :, ::-1])

        if fa_ret.img_ratoted_unified is not None:
            ax = fig.add_subplot(1,4,4)
            ax.set_axis_off()
            ax.set_title("rotated_unified")       
            ax.imshow(fa_ret.img_ratoted_unified[:, :, ::-1])
        
        plt.show()
