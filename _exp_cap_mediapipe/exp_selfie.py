from datetime import datetime 

def _check_mp_solution(mp_solution):
    pass

def _track_selfie(mp=None, cv2=None):
    if mp is None:
        import mediapipe as mp
    if cv2 is None:
        import cv2 as cv2        

    import numpy as np
    mp_drawing = mp.solutions.drawing_utils
    print(mp_drawing)
    
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    SFM_BG_COLOR = (192, 192, 192) # gray
    cap = cv2.VideoCapture(0)
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:

        _check_mp_solution(selfie_segmentation)

        bg_image = None
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = selfie_segmentation.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)
            if bg_image is None:
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = SFM_BG_COLOR
            output_image = np.where(condition, image, bg_image)

            cv2.imshow('MediaPipe Selfie Segmentation', output_image)
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

    _track_selfie(mp, cv2)


if __name__ == '__main__':
    do_exp()
