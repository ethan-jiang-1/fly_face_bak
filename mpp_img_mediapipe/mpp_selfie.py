from datetime import datetime
import cv2
import os
import numpy as np

class MppSelfieMarker():
    @classmethod
    def mark_selfie(cls, mp, slt_selfie, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        d0 = datetime.now()
        results = slt_selfie.process(image)
        dt = datetime.now() - d0
        print("inference time {:.3f}".format(dt.total_seconds()))

        # Draw the face mesh annotations on the image.
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

        BG_COLOR = (192, 192, 192) # gray
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
        print(output_image.shape, output_image.dtype)
        return output_image, dt

    @classmethod
    def create_slt_selfie(cls, mp):
        from utils_inspect.inspect_solution import inspect_solution
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        fslt_selfie = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) 
        inspect_solution(fslt_selfie)
        return fslt_selfie

    @classmethod
    def close_slt_selfie(cls, fslt_selfie):
        fslt_selfie.close()


def _find_all_images(src_dir):
    if not os.path.isdir(src_dir):
        raise ValueError("{} not folder".format(src_dir))

    names = os.listdir(src_dir)
    filenames = []
    for name in names:
        if name.endswith(".jpg") or name.endswith(".jpeg"):
            filenames.append(os.sep.join([src_dir, name]))
    return filenames

def _mark_selfie_imgs(src_dir, mp=None):
    from utils_inspect.inspect_mp import inspect_mp
    if mp is None:
        import mediapipe as mp
    inspect_mp(mp)

    filenames = _find_all_images(src_dir)

    slt_selfie = MppSelfieMarker.create_slt_selfie(mp)
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        slt_selfie.reset()
        image, _ = MppSelfieMarker.mark_selfie(mp, slt_selfie, image)
        if image is None:
            print("not able to mark landmark on", filename)

        #img_flip = cv2.flip(image, 1)
        dst_pathname = "{}{}{}".format(os.path.basename(src_dir) + "_output", os.sep, os.path.basename(filename).replace(".jp", "_sf.jp"))
        cv2.imwrite(dst_pathname, image)
        print("{} saved".format(dst_pathname))

    MppSelfieMarker.close_slt_selfie(slt_selfie)
    print("done")

def _get_root_dir():
    import os 

    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    return dir_root

def _add_root_in_sys_path():
    import sys 

    dir_root = _get_root_dir()
    if dir_root not in sys.path:
        sys.path.append(dir_root)

def do_exp():
    d0 = datetime.now()
    print("loading mediapipe...")
    import mediapipe as mp
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
   
    _mark_selfie_imgs(src_dir, mp)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
