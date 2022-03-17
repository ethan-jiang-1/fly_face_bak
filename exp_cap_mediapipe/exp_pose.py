from datetime import datetime 

def _track_pose(mp=None, cv2=None):
    if mp is None:
        import mediapipe as mp
    if cv2 is None:
        import cv2 as cv2        

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    #import numpy as np
    mp_drawing = mp.solutions.drawing_utils
    print(mp_drawing)
    
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def do_exp():
    d0 = datetime.now()
    print("loading mediapipe...")
    import mediapipe as mp
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    d0 = datetime.now()
    import cv2 as cv2
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    _track_pose(mp, cv2)


if __name__ == '__main__':
    do_exp()
