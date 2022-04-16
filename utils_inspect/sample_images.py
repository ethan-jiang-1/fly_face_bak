import os

class SampleImages():
    @classmethod
    def get_sample_images(cls, mid_name=None):
        dir_this = os.path.dirname(__file__)
        dir_imgs = os.sep.join([dir_this, "_sample_imgs"])
        names = os.listdir(dir_imgs)

        image_pathnames = []
        for name in names:
            if mid_name is not None:
                if not name.find(mid_name) != -1:
                    continue
            if name.endswith(".jpg") or name.endswith(".jpeg"):
                image_pathnames.append(os.sep.join([dir_imgs, name]))
        image_pathnames = sorted(image_pathnames)
        return image_pathnames 

    @classmethod
    def get_sample_images_hsi(cls):
        return cls.get_sample_images("_hsi_")

    @classmethod
    def get_sample_images_ctn(cls):
        return cls.get_sample_images("_ctn_")

    @classmethod
    def get_sample_images_icl(cls):
        return cls.get_sample_images("_icl_")

    @classmethod
    def get_sample_images_brd(cls):
        return cls.get_sample_images("_brd_")

    @classmethod
    def get_sample_images_female(cls):
        return cls.get_sample_images("F_")

    @classmethod
    def get_sample_images_male(cls):
        return cls.get_sample_images("M_")

if __name__ == '__main__':
    paths = SampleImages.get_sample_images_hsi()
    for idx, path in enumerate(paths):
        print(idx, path)
