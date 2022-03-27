import os
import cv2

class FileOp(object):
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
        output_dir = "_reserved_output_{}".format(os.path.basename(src_dir))
        os.makedirs(output_dir, exist_ok=True)
        dst_pathname = "{}{}{}".format(output_dir, os.sep, os.path.basename(filename).replace(".jp", "_{}.jp".format(suffix)))
        cv2.imwrite(dst_pathname, image)
        print("{} saved".format(dst_pathname))
