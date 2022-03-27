from collections import namedtuple
import math
import cv2
import numpy as np
from PIL import Image as PilImage
import matplotlib.pyplot as plt
from datetime import datetime 
from abc import ABC, abstractmethod

FA_IMG_SIZE = 512
FA_RESULT = namedtuple('FA_RESULT', 'img_org img_face img_rotated img_ratoted_unified dt')
FA_TRANSINFO = namedtuple("FA_TRANSINFO", 'img_org_shape bbox_crop img_crop_shape direction angle center_of_eyes distance_of_eyes left_eye_center right_eye_center point_3rd')

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

class ImgTransformAgent():
    @classmethod
    def _compute_eyes_location(cls, left_eye, right_eye):
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

    @classmethod
    def cal_transform_info(cls, img_org, bbox_crop_in_org, img_crop, left_eye_in_crop, right_eye_in_crop):
        left_eye_center, right_eye_center, center_of_eyes = cls._compute_eyes_location(left_eye_in_crop, right_eye_in_crop) 
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
        
        da = euclidean_distance(left_eye_center, point_3rd)
        db = euclidean_distance(right_eye_center, point_3rd)
        dc = euclidean_distance(right_eye_center, left_eye_center)

        cos_a = (db*db + dc*dc - da*da)/(2*db*dc)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi
        if direction == -1:
            angle = 90 - angle

        fti = FA_TRANSINFO(img_org_shape=img_org.shape,
                           bbox_crop=bbox_crop_in_org, 
                           img_crop_shape=img_crop.shape, 
                           direction=direction, 
                           angle=angle,
                           center_of_eyes=center_of_eyes,
                           distance_of_eyes=dc,
                           left_eye_center=left_eye_center, 
                           right_eye_center=right_eye_center, 
                           point_3rd=point_3rd)
        return fti

    @classmethod
    def plot_transinfo_keypoints(cls, img_crop, fti):
        cv2.circle(img_crop, fti.left_eye_center, 2, (255, 0, 0) , 2)
        cv2.circle(img_crop, fti.right_eye_center, 2, (255, 255, 0) , 2)
        cv2.circle(img_crop, fti.center_of_eyes, 2, (255, 0, 255) , 2)

        cv2.circle(img_crop, fti.point_3rd, 2, (255, 0, 0) , 2)
        
        cv2.line(img_crop,fti.right_eye_center, fti.left_eye_center,(67,67,67),1)
        cv2.line(img_crop,fti.left_eye_center, fti.point_3rd,(67,67,67),1)
        cv2.line(img_crop,fti.right_eye_center, fti.point_3rd,(67,67,67),1)

    @classmethod
    def tranform_to_img_rotated(cls, img_org, fti):
        direction, angle = fti.direction, fti.angle
        
        image_rotated = PilImage.fromarray(img_org)
        image_rotated = image_rotated.rotate(direction * angle)
        img_rotated = np.array(image_rotated)
        return img_rotated

    @classmethod
    def _rotate_cv_by_center(cls, image, angle, center = None, scale = 1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    @classmethod
    def _cal_src_dst_box(cls, delta_center_n, shape_org_n, shape_moved_n):
        if delta_center_n > 0:
            sn0 = delta_center_n
            snw = shape_org_n - delta_center_n
            dn0 = 0
            dnw = shape_moved_n
        elif delta_center_n < 0:
            sn0 = 0
            snw = shape_org_n
            dn0 = -delta_center_n
            dnw = shape_moved_n - dn0
        else:
            sn0 = 0
            snw = shape_org_n
            dn0 = 0
            dnw = shape_moved_n           
        minw = min(snw, dnw)
        dnw, snw = minw, minw
        return sn0, snw, dn0, snw

    @classmethod
    def _get_moved_param(cls, img_org_shape, center_of_eyes, debug=False):
        shape_height_org, shape_width_org = img_org_shape[0], img_org_shape[1]

        center_of_img = (int(shape_height_org / 2) , int(shape_width_org / 2))
        center_of_eyes_yx = (center_of_eyes[1], center_of_eyes[0])

        delta_center_y = int(center_of_eyes_yx[0] - center_of_img[0])
        delta_center_x = int(center_of_eyes_yx[1] - center_of_img[1])

        shape_height_moved = int(shape_height_org + delta_center_y)
        shape_width_moved = int(shape_width_org + delta_center_x)
        img_moved_shape = (shape_height_moved, shape_width_moved, img_org_shape[2])

        sx0, sxw, dx0, dxw = cls._cal_src_dst_box(delta_center_x, shape_width_org, shape_width_moved)
        sy0, syh, dy0, dyh = cls._cal_src_dst_box(delta_center_y, shape_height_org, shape_height_moved)

        if debug:
            print()
            print("center_of_img_yx: ", center_of_img)
            print("center_of_eyes_yx:", center_of_eyes_yx)
            print("img_org_shape:  ", img_org_shape)
            print("img_moved_shape:", img_moved_shape)
            print("move src        ", (sy0, syh, sx0, sxw))
            print("move dst        ", (dy0, dyh, dx0, dxw))
            print()
        return (dy0, dyh, dx0, dxw), (sy0, syh, sx0, sxw), img_moved_shape

    @classmethod
    def transfer_to_img_unified(cls, img_org, fti, debug=False):
        bbox_crop = fti.bbox_crop
        center_crop = fti.center_of_eyes

        center_of_eyes = (center_crop[0] + bbox_crop[0], center_crop[1] + bbox_crop[1])
        angle = fti.direction * fti.angle
        img_rotated_by_eye_center = cls._rotate_cv_by_center(img_org, angle, center=center_of_eyes)
        print('img_rotated_by_eye_center.shape', img_rotated_by_eye_center.shape)

        (dy0, dyh, dx0, dxw), (sy0, syh, sx0, sxw), img_moved_shape = cls._get_moved_param(img_rotated_by_eye_center.shape, center_of_eyes, debug=debug)
        img_moved_rotated_center = np.zeros(img_moved_shape, dtype=img_org.dtype)
        img_src = img_rotated_by_eye_center[sy0:sy0+syh, sx0:sx0+sxw]
        img_moved_rotated_center[dy0:dy0+dyh, dx0:dx0+dxw] = img_src      

        print("img_moved_rotated_center.shape", img_moved_rotated_center.shape)

        return img_moved_rotated_center
        

class FaceAligmentBase(ABC):
    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def create_detector():
        pass

    @abstractmethod
    def close_detector():
        pass

    @abstractmethod
    def detect_face_and_eyes(self, img_org):
        pass

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
        fti = ImgTransformAgent.cal_transform_info(img_org, bbox_crop_in_org, img_crop, left_eye_in_crop, right_eye_in_crop)
        if self.debug:
            ImgTransformAgent.plot_transinfo_keypoints(img_crop, fti)

        img_rotated = ImgTransformAgent.tranform_to_img_rotated(img_raw, fti)

        img_ratoted_unified = ImgTransformAgent.transfer_to_img_unified(img_raw, fti, debug=self.debug)

        dt = dt=datetime.now() - d0
        print("inference time: {:.3f}".format(dt.total_seconds()))
        return FA_RESULT(img_org=img_org, 
                         img_face=img_crop, 
                         img_rotated=img_rotated, 
                         img_ratoted_unified=img_ratoted_unified,
                         dt=dt)

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
            ax.set_title("crop_face")
            ax.set_axis_off()
            ax.imshow(fa_ret.img_face)

        if fa_ret.img_rotated is not None:
            ax = fig.add_subplot(1,4,3)
            ax.set_axis_off()
            ax.set_title("rotated")
            ax.imshow(fa_ret.img_rotated[:, :, ::-1])

        if fa_ret.img_ratoted_unified is not None:
            ax = fig.add_subplot(1,4,4)
            #ax.set_axis_off()
            ax.set_title("crop_rotated_unified")       
            ax.imshow(fa_ret.img_ratoted_unified[:, :, ::-1])
        
        plt.show()
