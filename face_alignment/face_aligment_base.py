from collections import namedtuple
import math
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime 


FA_RESULT = namedtuple('FA_RESULT', 'img_org img_face img_rotated img_ratoted_face dt')
FA_IMG_SIZE = 512

class FaceAligmentBase(object):
    def __init__(self, debug=False):
        self.debug = debug

    def _find_rotate_info(self, img_face, left_eye, right_eye):
        #center of eyes
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        center_of_eyes = (int((left_eye_x+right_eye_x)/2), int((left_eye_y+right_eye_y)/2))
        
        if self.debug:
            cv2.circle(img_face, left_eye_center, 2, (255, 0, 0) , 2)
            cv2.circle(img_face, right_eye_center, 2, (255, 255, 0) , 2)
            cv2.circle(img_face, center_of_eyes, 2, (255, 0, 255) , 2)
        
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
            cv2.circle(img_face, point_3rd, 2, (255, 0, 0) , 2)
            
            cv2.line(img_face,right_eye_center, left_eye_center,(67,67,67),1)
            cv2.line(img_face,left_eye_center, point_3rd,(67,67,67),1)
            cv2.line(img_face,right_eye_center, point_3rd,(67,67,67),1)
        
        a = self.euclidean_distance(left_eye_center, point_3rd)
        b = self.euclidean_distance(right_eye_center, point_3rd)
        c = self.euclidean_distance(right_eye_center, left_eye_center)
        
        #print("left eye: ", left_eye_center)
        #print("right eye: ", right_eye_center)
        #print("additional point: ", point_3rd)
        #print("triangle lengths: ",a, b, c)
        
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        #print("cos(a) = ", cos_a)
        angle = np.arccos(cos_a)
        #print("angle: ", angle," in radian")
        
        angle = (angle * 180) / math.pi
        print("angle: ", angle," in degree")
        
        if direction == -1:
            angle = 90 - angle
        
        print("angle: ", angle," in degree")
        return direction, angle

    def align_face(self, img_org):
        d0 = datetime.now()
        img_raw = img_org.copy()
        
        img_face, left_eye, right_eye = self.detect_face_and_eys(img_org.copy())

        if img_face is None:
            return FA_RESULT(img_org=img_org, 
                              img_face=None, 
                              img_rotated=None, 
                              img_ratoted_face=None,
                              dt=datetime.now() - d0) 

        if left_eye is None and right_eye is None:
            return FA_RESULT(img_org=img_org, 
                              img_face=img_face, 
                              img_rotated=None, 
                              img_ratoted_face=None,
                              dt=datetime.now() - d0) 
        
        #rotate image
        direction, angle = self._find_rotate_info(img_face, left_eye, right_eye)
        
        img_rotated = Image.fromarray(img_raw)
        img_rotated = np.array(img_rotated.rotate(direction * angle))
        img_ratoted_face = self.detect_face(img_rotated)

        dt = dt=datetime.now() - d0
        print("inference time: {:.3f}".format(dt.total_seconds()))
        return FA_RESULT(img_org=img_org, 
                         img_face=img_face, 
                         img_rotated=img_rotated, 
                         img_ratoted_face=img_ratoted_face,
                         dt=dt)
                    
    def detect_face_and_eys(self, img_org):
        raise ValueError("not-implemented")
        #return None, None  # img_face, eyes (iris)

    def detect_face(self, img_org):
        raise ValueError("not-implemented")
        # return None

    def euclidean_distance(self, a, b):
        x1 = a[0]; y1 = a[1]
        x2 = b[0]; y2 = b[1]
        
        return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

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

        if fa_ret.img_ratoted_face is not None:
            ax = fig.add_subplot(1,4,4)
            ax.set_axis_off()
            ax.set_title("rotated_face")       
            ax.imshow(fa_ret.img_ratoted_face[:, :, ::-1])
        
        plt.show()
