import numpy as np
#import math
import cv2

FMB_FILL_COLOR = (216, 216, 216)
FMB_SELFIE_FILL_COLOR = (192, 192, 192)

class FcxBeardNew():
    @classmethod
    def process_img(cls, image, mesh_results, debug=False):
        img_beard_region = cls._clean_region(image)

        img_gabor_filtered = cls._filter_by_gabor_filter(img_beard_region)

        img_beard_color_blur = cv2.blur(img_gabor_filtered, (5,5))
        img_beard_gray = cv2.cvtColor(img_beard_color_blur, cv2.COLOR_BGR2GRAY)
        
        img_beard_black = img_beard_gray
        img_beard_white = cv2.bitwise_not(img_beard_black)

        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image, img_beard_region, img_gabor_filtered, img_beard_black, img_beard_white],
                                 names=["org", "flood", "gabor", "black", "white"])            

        print(img_beard_white.shape, img_beard_white.dtype)
        return img_beard_white

    @classmethod
    def _clean_region(cls, image):
        image_flood = image.copy()
        cls._flood_fill(image_flood, pt_seed=(0,0), color_fill=FMB_SELFIE_FILL_COLOR)
        return image_flood

    @classmethod
    def _filter_by_gabor_filter(cls, image):
        ksize = 11  # kernal size
        lamda = np.pi / 2.0  # length of wave
        theta = np.pi * 0.5  # 90 degree
        #theta = np.pi * 0.0  # 0 degree
        kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        gabor_filters = kern 

        img_accum = np.zeros_like(image)
        for gabor_filter in gabor_filters:
            img_fimg = cv2.filter2D(image, cv2.CV_8UC1, gabor_filter)
            img_accum = np.maximum(img_accum, img_fimg, img_accum)
        return img_accum

    @classmethod
    def _flood_fill(cls, image, pt_seed=(0, 0), color_fill=FMB_FILL_COLOR):
        h, w = image.shape[:2]
        mask_flood = np.zeros([h+2, w+2],np.uint8)
        cv2.floodFill(image, mask_flood, pt_seed, color_fill, flags=cv2.FLOODFILL_FIXED_RANGE)
