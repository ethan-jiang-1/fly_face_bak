
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

class PlotHelper(object):
    @classmethod
    def plot_imgs(cls, imgs, names=None, title=None):
        fig = plt.figure(figsize=(14, 6))    
        if title is not None:
            ax = plt.gca()
            ax.set_axis_off()
            ax.set_title(title)

        num = len(imgs)
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(1, num, i + 1)

            if names is not None:
                ax.set_title(names[i])
            if img is not None:
                ax.imshow(img, interpolation="none")
        plt.show()    

def _get_entopy_masked(img_org, disk_num=8, th_val=192, fill_gray=192):
    from skimage.filters.rank import entropy
    from skimage.morphology import disk

    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    img_entr = entropy(img_gray, disk(disk_num))

    img_mask = img_entr.copy()
    img_mask -= img_mask.min()
    img_mask /= img_mask.max()
    img_mask *= 255

    img_mask_u8 = img_mask.astype(np.uint8)
    _, img_mask_u8 = cv2.threshold(img_mask_u8, th_val, 255, cv2.THRESH_BINARY)

    img_mask_mono = img_mask_u8.copy()
    img_mask_mono[np.where(img_mask_mono[:,:] == 0)] = fill_gray
   
    img_masked_filled = cv2.bitwise_and(img_org, img_org, mask = img_mask_u8)   
    img_mask_color = cv2.cvtColor(img_mask_mono, cv2.COLOR_GRAY2RGB)
    img_masked_filled += img_mask_color
    return img_masked_filled, img_entr, img_mask_u8 


def do_exp(full_path):
    img_org = cv2.imread(full_path, cv2.IMREAD_ANYCOLOR)

    img_masked_filled_4, img_entr_4, img_mask_4  = _get_entopy_masked(img_org, disk_num=4)
    img_masked_filled_8, img_entr_8, img_mask_8  = _get_entopy_masked(img_org, disk_num=8)
 
    PlotHelper.plot_imgs([img_org, img_masked_filled_4, img_entr_4, img_mask_4 , img_masked_filled_8, img_entr_8, img_mask_8])


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    filename = "dataset_org_beard_styles/Beard Version 1.1/01/01_010.jpg"

    full_path = parent_dir + os.sep + filename

    do_exp(full_path)

