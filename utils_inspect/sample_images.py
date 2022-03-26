import os

def get_sample_images(name_prefix=None):
    dir_this = os.path.dirname(__file__)
    dir_imgs = os.sep.join([dir_this, "_sample_imgs"])
    names = os.listdir(dir_imgs)

    image_pathnames = []
    for name in names:
        if name_prefix is not None:
            if not name.startswith(name_prefix):
                continue
        if name.endswith(".jpg") or name.endswith(".jpeg"):
            image_pathnames.append(os.sep.join([dir_imgs, name]))
    image_pathnames = sorted(image_pathnames)
    return image_pathnames 

def get_sample_images_hsi():
    return get_sample_images("hsi_")

def get_sample_images_ctn():
    return get_sample_images("ctn_")


if __name__ == '__main__':
    paths = get_sample_images_hsi()
    for idx, path in enumerate(paths):
        print(idx, path)
