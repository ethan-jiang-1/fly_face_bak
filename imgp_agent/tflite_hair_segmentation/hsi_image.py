import os
import cv2

class HsiReseredImg():
    @classmethod
    def get_resvered_img_pathname(cls, index):
        dir_this = os.path.dirname(__file__)
        dir_parent = os.path.dirname(dir_this)
        dir_root = os.path.dirname(dir_parent)
        img_pathname = os.sep.join([dir_root, "utils_inspect", "_sample_imgs", "hsi_image{}.jpeg".format(index)])
        if os.path.isfile(img_pathname):
            return img_pathname
        return None


class HsiImageAlignment():
    @classmethod
    def convert_to_aligned_img_cv512(cls, img_pathname, img_size=512):
        img = cv2.imread(img_pathname, cv2.IMREAD_ANYCOLOR)
        img_resized = cv2.resize(img, (img_size, img_size))
        return img_resized


if __name__ == '__main__':
    img_path = HsiReseredImg.get_resvered_img_pathname(1)
    print(img_path)
    img_resized = HsiImageAlignment.convert_to_aligned_img_cv512(img_path)
    print(img_resized.shape)
