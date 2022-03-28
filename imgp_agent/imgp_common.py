import os
import cv2
import matplotlib.pyplot as plt

class FileHelper(object):
    @classmethod
    def find_all_images(cls, src_dir):
        if not os.path.isdir(src_dir):
            raise ValueError("{} not folder".format(src_dir))

        names = os.listdir(src_dir)
        filenames = []
        for name in names:
            if name.endswith(".jpg") or name.endswith(".jpeg"):
                filenames.append(os.sep.join([src_dir, name]))
        return filenames

    @classmethod
    def save_output_image(cls, image, src_dir, filename, suffix):
        if image is None:
            return False
        output_dir = "_reserved_output_{}".format(os.path.basename(src_dir))
        os.makedirs(output_dir, exist_ok=True)
        dst_pathname = "{}{}{}".format(output_dir, os.sep, os.path.basename(filename).replace(".jp", "_{}.jp".format(suffix)))
        cv2.imwrite(dst_pathname, image)
        if not os.path.isfile(dst_pathname):
            print("ERROR, failed to save {}".format(dst_pathname))
            return False
        print("{} saved".format(dst_pathname))
        return True

class PlotHelper(object):
    @classmethod
    def plot_img(cls, img, name=None):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        if name is not None:
            ax.set_title(name)
        ax.imshow(img)
        plt.show()

    @classmethod
    def plot_imgs(cls, imgs, names=None, title=None, horizontal=True):
        if horizontal:
            fig = plt.figure(figsize=(14, 6))    
        else:
            fig = plt.figure(figsize=(6, 12))
        if title is not None:
            ax = plt.gca()
            ax.set_axis_off()
            ax.set_title(title)

        num = len(imgs)
        for i, img in enumerate(imgs):
            if horizontal:
                ax = fig.add_subplot(1, num, i + 1)
            else:
                ax = fig.add_subplot(num, 1, i + 1)

            if names is not None:
                ax.set_title(names[i])
            ax.imshow(img)
        plt.show()    
