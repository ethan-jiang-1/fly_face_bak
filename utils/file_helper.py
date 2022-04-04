import os
import cv2

class FileHelper(object):
    @classmethod
    def find_all_images(cls, src_dir):
        if not os.path.isdir(src_dir):
            raise ValueError("{} not folder".format(src_dir))

        names = os.listdir(src_dir)
        filenames = []
        for name in names:
            if cls.is_supported_img_filename(name):
                filenames.append(os.sep.join([src_dir, name]))                
        filenames = sorted(filenames)
        return filenames

    @classmethod
    def is_supported_img_filename(cls, filename):
        name = filename.lower()
        if name.endswith(".jpg") or name.endswith(".jpeg"): # or name.endswith(".png"):
            return True
        return False

    @classmethod
    def save_output_image(cls, image, src_dir, filename, suffix, output_dir=None):
        if image is None:
            return False
        if output_dir is None:
            output_dir = "_reserved_output_{}".format(os.path.basename(src_dir))
    
        os.makedirs(output_dir, exist_ok=True)
        dst_pathname = "{}{}{}".format(output_dir, os.sep, os.path.basename(filename).replace(".jp", "_{}.jp".format(suffix)))
        cv2.imwrite(dst_pathname, image)
        if not os.path.isfile(dst_pathname):
            print("ERROR, failed to save {}".format(dst_pathname))
            return False
        print("{} saved".format(dst_pathname))
        return True
