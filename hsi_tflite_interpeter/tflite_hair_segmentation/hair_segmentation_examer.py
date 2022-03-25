import cv2
import os

from hair_segmentation_interpeter import HairSegmentationMaskSharpener
from hair_segmentation_interpeter import HairSegmentationInterpreter

class HsiOutputHelper(object):
    @classmethod
    def get_ignored_pathname(cls, filename, img_sub_folder="_ignored_imgs"):
        dir_this = os.path.dirname(__file__)
        dir_parent = os.path.dirname(dir_this)
        img_dir = os.sep.join([dir_parent, img_sub_folder])
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.sep.join([img_dir, filename])
        return img_path

    @classmethod
    def save_hsi_result(cls, hsi_ret):
        img_name = hsi_ret.img_name
        
        mbs_pathname = cls.get_ignored_pathname("{}-mbs.jpg".format(img_name.replace(".","_")))
        mbs_img = HairSegmentationMaskSharpener.make_grayscale_img_u8(hsi_ret.mask_black_sharp)
        cv2.imwrite(mbs_pathname, mbs_img)
        print("{} saved".format(mbs_pathname))

        mb_pathname = cls.get_ignored_pathname("{}-mb.jpg".format(img_name.replace(".","_")))
        mb_img = HairSegmentationMaskSharpener.make_grayscale_img_u8(hsi_ret.mask_black)
        cv2.imwrite(mb_pathname, mb_img)
        print("{} saved".format(mb_pathname))

    @classmethod
    def save_hsi_results(cls, hsi_rets):
        for hsi_ret in hsi_rets:
            cls.save_hsi_result(hsi_ret)

    @classmethod
    def display_hsi_results(cls, hsi_rets):
        import matplotlib.pyplot as plt

        num = len(hsi_rets)
        print("plt images: {}".format(num))
        if num == 1:
            fig = plt.figure(figsize=(16, 6))
        elif num == 2:
            fig = plt.figure(figsize=(16, 8))
        else:
            fig = plt.figure(figsize=(8, 8))

        for i in range(num):
            ret = hsi_rets[i]
            ax = fig.add_subplot(num, 5, i * 5 + 1)
            ax.set_axis_off()
            ax.imshow(ret.img_unified)

            ax = fig.add_subplot(num, 5, i * 5 + 2)
            ax.set_axis_off()
            ax.imshow(ret.mask_black)
            ax = fig.add_subplot(num, 5, i * 5 + 3)
            ax.set_axis_off()
            ax.imshow(ret.mask_black_sharp)

            ax = fig.add_subplot(num, 5, i * 5 + 4)
            ax.set_axis_off()
            ax.imshow(ret.mask_white)
            ax = fig.add_subplot(num, 5, i * 5 + 5)
            ax.set_axis_off()
            ax.imshow(ret.mask_white_sharp)

        plt.show()
        print("plt show...")


class HsiExamer():
    @classmethod
    def debug_single_image(cls, img_pathname, show=True):
        hsi = HairSegmentationInterpreter(debug=True)
        hsi.validate()

        img = hsi.get_resized_img(img_pathname)
        hsi_ret  = hsi.process_img_cv512(img, img_name=os.path.basename(img_pathname))
        if show:
            HsiOutputHelper.display_hsi_results([hsi_ret])
        else:
            HsiOutputHelper.save_hsi_results([hsi_ret])

    @classmethod
    def debug_multiple_images(cls, img_pathnames, show=True):
        hsi = HairSegmentationInterpreter(debug=True)
        hsi.validate()

        hsi_rets = []
        for img_pathname in img_pathnames:
            img = hsi.get_resized_img(img_pathname)
            ret = hsi.process_img_cv512(img, img_name=os.path.basename(img_pathname))
            hsi_rets.append(ret)

        if show:
            HsiOutputHelper.display_hsi_results(hsi_rets)
        else:
            HsiOutputHelper.save_hsi_results(hsi_rets)

    @classmethod
    def get_resvered_img_pathname(cls, index):
        dir_this = os.path.dirname(__file__)
        dir_parent = os.path.dirname(dir_this)
        img_pathname = os.sep.join([dir_parent, "_reserved_imgs", "image{}.jpeg".format(index)])
        if os.path.isfile(img_pathname):
            return img_pathname
        return None

    @classmethod
    def do_debug_reseved_img(cls, index, show=True):
        img_pathname = cls.get_resvered_img_pathname(index)

        cls.debug_single_image(img_pathname, show=show)

    @classmethod
    def do_debug_reseved_imgs(cls,indexs, show=True):
        img_pathnames = []
        for index in indexs:
            img_pathname = cls.get_resvered_img_pathname(index)
            if img_pathname is not None:
                img_pathnames.append(img_pathname)

        cls.debug_multiple_images(img_pathnames, show=show)


def _add_parent_in_sys_path():
    import sys
    dir_this = os.path.dirname(__file__)
    dir_parent = os.path.dirname(dir_this)
    if dir_parent not in sys.path:
        sys.path.append(dir_parent)


if __name__ == '__main__':
    _add_parent_in_sys_path()
    #HsiExamer.do_debug_reseved_img(3)
    #HsiExamer.do_debug_reseved_imgs([1, 3], show=True)
    #HsiExamer.do_debug_reseved_imgs([1, 2, 3, 4], show=True)
    #HsiExamer.do_debug_reseved_imgs([5, 6, 7, 8], show=True)
    HsiExamer.do_debug_reseved_imgs([1, 2, 3, 4, 5, 6, 7, 8], show=False)
