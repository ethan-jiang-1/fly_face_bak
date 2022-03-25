import os
import cv2
import numpy as np
from datetime import datetime 
from collections import namedtuple

HSI_RESULT = namedtuple('HSI_RESULT', 'img_unified mask_white mask_black mask_white_sharp mask_black_sharp dt img_name')
HSI_IMG_SIZE = 512

class HairSegmentationInterpreter(object):
    def __init__(self, debug=False):
        self.debug = debug
        self.ipt = self._create_tfl_interperter()

    def _create_tfl_interperter(self):
        try:
            import tflite_runtime.interpreter as tflite
            self.tflite = tflite
            model_pathname = self._get_model_pathname()
            ipt = tflite.Interpreter(model_path=model_pathname)
            ipt.allocate_tensors()
            return ipt
        except Exception as ex:
            print("Exception occured", ex)
            return None

    def _get_model_pathname(self):
        this_dir = os.path.dirname(__file__)
        paths = [this_dir, "hair_segmentation.tflite"]
        full_path = os.sep.join(paths)
        if not os.path.isfile(full_path):
            raise ValueError("{} not exist".format(full_path))
        return full_path

    def validate(self):
        if self.ipt is None:
            return False
        if self.debug:
            self.dump_tflite_info()
            self.dump_ipt_info() 
        return True

    def dump_tflite_info(self):
        print("== TFLITE runtime ==")
        print("", "location: ", self.tflite.__file__)
        print("", "package:  ", self.tflite.__package__)
        print()

    def _dump_tflite_tensor(self, tensor_dp):
        print("", "index:  ", tensor_dp['index'])
        print("", "name:   ", tensor_dp['name'])
        print("", "shape:  ", tensor_dp['shape'])
        print("", "type:   ", tensor_dp['dtype'])
        print("", "shape_s:", tensor_dp["shape_signature"])

    def dump_ipt_info(self):
        print("== Input details ==")
        inputs = self.ipt.get_input_details()
        for idx in range(len(inputs)):
            input = inputs[idx]
            print("DUMP INPUT", idx) # , input)
            self._dump_tflite_tensor(input)
        print()

        print("== Output details ==")
        outputs = self.ipt.get_output_details()
        for idx in range(len(outputs)):
            output = outputs[idx]
            print("DUMP OUTPUT", idx) # , output)
            self._dump_tflite_tensor(output)
        print()

    def get_resized_img(self, img_pathname):
        img = cv2.imread(img_pathname, cv2.IMREAD_ANYCOLOR)
        #print(img.shape)
        img_size = HSI_IMG_SIZE
        img_resized = cv2.resize(img, (img_size, img_size))
        #print(img_resized.shape)
        return img_resized

    def process_img_cv512(self, img_cv512, img_name=None):
        if self.ipt is None:
            raise ValueError("Interperter not found")

        img_size = HSI_IMG_SIZE

        if img_cv512.shape[0] != img_size and img_cv512.shape[1] != img_size and img_cv512.shape[2] != 3:
            raise ValueError("expecting (512, 512, 3) instead of {}".format(img_cv512))

        self.ipt.reset_all_variables()

        d0 = datetime.now()
        img_unified = img_cv512.copy() 
        img_unified = img_unified / 256

        img_tfl512 = np.zeros((1, img_size, img_size, 4), dtype=np.float32)    
        img_tfl512[0, :, :, 0:3] = img_unified

        inputs = self.ipt.get_input_details()
        idx_input = inputs[0]['index']
        self.ipt.set_tensor(idx_input, img_tfl512)   

        self.ipt.invoke()

        outputs = self.ipt.get_output_details()
        idx_output = outputs[0]['index']
        result_masks = self.ipt.get_tensor(idx_output)

        mask_black = np.zeros((img_size, img_size, 1), dtype=np.int32)
        mask_white = np.zeros((img_size, img_size, 1), dtype=np.float32)

        mask_black = result_masks[0, :, :, 0]
        mask_white = result_masks[0, :, :, 1]

        d1 = datetime.now()
        dt = d1 - d0
        print("inference time: {:.3f}sec for {}".format(dt.total_seconds(), img_name))

        mask_black_sharp = HairSegmentationMaskSharpener.sharpen_mask_black(mask_black)
        mask_white_sharp = HairSegmentationMaskSharpener.sharpen_mask_white(mask_white)

        result = HSI_RESULT(img_unified=img_unified, 
                            mask_white=mask_white,
                            mask_black=mask_black,
                            mask_white_sharp=mask_white_sharp, 
                            mask_black_sharp=mask_black_sharp,
                            dt=dt, 
                            img_name=img_name)

        return result

class HairSegmentationMaskSharpener(object):
    MBLUR_KERNAL = 13
    BLUR_KERNAL1 = (15, 15)
    BLUR_KERNAL2 = (7, 7)

    @classmethod
    def make_grayscale_img_u8(cls, mask_float):
        if mask_float.dtype == np.uint8:
            return mask_float

        mask = mask_float.copy()
        max = mask.max()
        min = mask.min()

        mask[np.where(mask[:,:] >= min)] += -min 
        mask[np.where(mask[:,:] >= 0)] *= 254 / (max - min)
        mask = mask.reshape((512, 512, 1))
        mask_u8 = np.uint8(mask)
        return mask_u8

    @classmethod
    def _sharpen_mask_u8(cls, mask_u8, mode="MIXED", fallback_val=0):
        if mode == "ADAPTIVE":
            mask_thresh = cv2.adaptiveThreshold(mask_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif mode == "MIXED":
            mask_blur = cv2.medianBlur(mask_u8, cls.MBLUR_KERNAL)
            mask_blur = cv2.GaussianBlur(mask_blur, cls.BLUR_KERNAL1, 0)
            mask_blur = cv2.GaussianBlur(mask_blur, cls.BLUR_KERNAL2, 0)
            _, mask_thresh = cv2.threshold(mask_blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(mask_thresh, 
                                            cv2.RETR_TREE, # cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) # 得到轮廓信息
            if len(contours) > 10:
                mask_thresh[:, :] = fallback_val

        return mask_thresh

    @classmethod
    def sharpen_mask_black(cls, mask_black, mode="MIXED"):
        mask_u8 = cls.make_grayscale_img_u8(mask_black)
        return cls._sharpen_mask_u8(mask_u8, mode=mode, fallback_val=0)

    @classmethod
    def sharpen_mask_white(cls, mask_white, mode="MIXED"):
        mask_u8 = cls.make_grayscale_img_u8(mask_white)
        return cls._sharpen_mask_u8(mask_u8, mode=mode, fallback_val=255)


if __name__ == '__main__':
    hsi = HairSegmentationInterpreter(debug=True)
    if not hsi.validate():
        raise ValueError("tflite runtime is not valid")
