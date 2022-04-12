from datetime import datetime
import cv2
import os
import numpy as np
import mediapipe as mp

SFM_FG_COLOR = (192, 192, 192)

#SFM_BG_COLOR = (192, 192, 192)
SFM_BG_COLOR = (0, 0, 0)
#SFM_BG_COLOR = (112, 112, 112)
#SFM_BG_COLOR = (64, 64, 64)

#SF_THRESHOLD = 0.1
SF_THRESHOLD = 0.3

class ImgpSelfieMarker():
    slt_selfie = None 
    slt_reference = 0
    
    @classmethod
    def init_imgp(cls):
        cls.slt_reference += 1
        if cls.slt_selfie is None:
            from utils_inspect.inspect_solution import inspect_solution
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            cls.slt_selfie = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) 
            inspect_solution(cls.slt_selfie)
            print("ImgpSelfieMarker inited")

    @classmethod
    def close_imgp(cls):
        cls.slt_reference -= 1
        if cls.slt_reference == 0:
            if cls.slt_selfie is not None:
                cls.slt_selfie.close()
                cls.slt_selfie = None
            print("ImgpSelfieMarker closed")

    @classmethod
    def fliter_selfie(cls, image, debug=False):
        if cls.slt_selfie is None:
            cls.init_imgp()
        
        slt_selfie = cls.slt_selfie
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        d0 = datetime.now()
        slt_selfie.reset()
        results = slt_selfie.process(image)
        dt = datetime.now() - d0
        print("inference time(sf) :{:.3f}".format(dt.total_seconds()))

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > SF_THRESHOLD
        if debug:
            print("condition.shape", condition.shape,  "conditoin.dtype", condition.dtype)
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)

        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = SFM_BG_COLOR
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = SFM_FG_COLOR

        output_image = np.where(condition, image, bg_image)
        if debug:
            print("output_image.shape", output_image.shape, "output_image.dtype", output_image.dtype)

        mask_image = np.where(condition, fg_image, bg_image)
        mask_image_wf = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        _, mask_image_wf = cv2.threshold(mask_image_wf, 128, 255, cv2.THRESH_BINARY)

        # if debug:
        #     from imgp_common import PlotHelper
        #     PlotHelper.plot_imgs([image, bg_image, fg_image, mask_image, mask_image_wf],
        #                           names=["image", "bg_image", "fg_image", "mask_image", "mask_image_wf"])

        return output_image, mask_image_wf


def _mark_selfie_imgs(src_dir, show=True):
    from imgp_agent.imgp_common import FileHelper, PlotHelper
    from utils.colorstr import log_colorstr
    filenames = FileHelper.find_all_images(src_dir)

    ImgpSelfieMarker.init_imgp()
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image, image_mask_wf = ImgpSelfieMarker.fliter_selfie(image, debug=True)
        if image is None:
            log_colorstr("red", "ERROR: not able to mark selfie on", filename)
        else:
            FileHelper.save_output_image(image, src_dir, filename, "selfie")
            FileHelper.save_output_image(image_mask_wf, src_dir, filename, "selfie_mask")
            if show:
                PlotHelper.plot_imgs([image, image_mask_wf], names=["selfie", "mask"])

    ImgpSelfieMarker.close_imgp()
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
    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    #src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
    src_dir = os.sep.join([_get_root_dir(), "utils_inspect", "_sample_imgs"])

    _mark_selfie_imgs(src_dir, show=True)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
