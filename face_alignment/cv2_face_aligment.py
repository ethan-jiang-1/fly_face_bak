import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from PIL import Image


#------------------------

#opencv path
def create_cv_detectors():
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

    face_detector = cv2.CascadeClassifier(face_detector_path)
    eye_detector = cv2.CascadeClassifier(eye_detector_path) 
    nose_detector = cv2.CascadeClassifier(nose_detector_path) 
    return face_detector, eye_detector, nose_detector

face_detector, eye_detector, nose_detector = create_cv_detectors()

#------------------------

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectFace(img):
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    #print("found faces: ", len(faces))

    if len(faces) > 0:
        face = faces[0]
        face_x, face_y, face_w, face_h = face
        img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img, img_gray
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_gray
        #raise ValueError("No face found in the passed image ")

def alignFace(img_path):
    img_org = cv2.imread(img_path)

    img_raw = img_org.copy()

    img_face, gray_img = detectFace(img_org)
    
    #eyes = eye_detector.detectMultiScale(gray_img)
    eyes = eye_detector.detectMultiScale(gray_img, 1.1, 10)
    eyes = sorted(eyes, key = lambda v: abs((v[0] - v[2]) * (v[1] - v[3])), reverse=True)
    eyes =np.array(eyes)
    
    #print("found eyes: ",len(eyes))
    
    rotated_img = None
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
        
        #--------------------
        #decide left and right eye
        
        eye_1 = eyes[0]; eye_2 = eyes[1]
        
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        
        #--------------------
        #center of eyes
        
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        #center_of_eyes = (int((left_eye_x+right_eye_x)/2), int((left_eye_y+right_eye_y)/2))
        
        cv2.circle(img_face, left_eye_center, 2, (255, 0, 0) , 2)
        cv2.circle(img_face, right_eye_center, 2, (255, 0, 0) , 2)
        #cv2.circle(img, center_of_eyes, 2, (255, 0, 0) , 2)
        
        #----------------------
        #find rotation direction
        
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 # rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 # rotate inverse direction of clock
            print("rotate to inverse clock direction")
        
        #----------------------
        
        cv2.circle(img_face, point_3rd, 2, (255, 0, 0) , 2)
        
        cv2.line(img_face,right_eye_center, left_eye_center,(67,67,67),1)
        cv2.line(img_face,left_eye_center, point_3rd,(67,67,67),1)
        cv2.line(img_face,right_eye_center, point_3rd,(67,67,67),1)
        
        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, point_3rd)
        c = euclidean_distance(right_eye_center, left_eye_center)
        
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
        
        #--------------------
        #rotate image
        
        rotated_img = Image.fromarray(img_raw)
        rotated_img = np.array(rotated_img.rotate(direction * angle))
    
    return img_org, img_face, rotated_img
    
#------------------------

def exam_face_aligment(face_crop=False):
    #test_set = ["angelina.jpg", "angelina2.jpg", "angelina3.jpg"]
    #test_set = ["angelina.jpg"]
    from utils_inspect.sample_images import get_sample_images
    test_set = get_sample_images()

    for instance in test_set:
        img_org, img_face, rotated_img = alignFace(instance)

        fig = plt.figure(figsize=(12, 6))

        ax = fig.add_subplot(1,4,1)
        ax.imshow(img_org)

        ax = fig.add_subplot(1,4,2)
        ax.imshow(img_face)

        if rotated_img is None:
            print("not able to align", instance)
            continue

        ax = fig.add_subplot(1,4,3)
        ax.imshow(rotated_img[:, :, ::-1])

        ax = fig.add_subplot(1,4,4)       
        img, gray_img = detectFace(rotated_img)
        ax.imshow(img[:, :, ::-1])
        
        plt.show()

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
