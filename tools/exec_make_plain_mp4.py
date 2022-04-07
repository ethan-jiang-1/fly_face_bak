#!/usr/bin/env python

#IMGS_FOLDER = "_gen_render_mcd_burger"
IMGS_FOLDER = "_gen_render_mcd_beard"


def gen_mp4():
    import os
    from bld_gen.tools.mp4_maker import Mp4Maker

    dir_root = _get_root_dir()

    src_dir = "{}/{}".format(dir_root, IMGS_FOLDER)
    dst_dir = "{}/_gen_render_mp4".format(dir_root)
    os.makedirs(dst_dir, exist_ok=True)

    src_name = IMGS_FOLDER.replace("_gen_render", "")

    filenames = Mp4Maker.get_sorted_img_files(src_dir)
    mp4_filename = Mp4Maker.create_mp4_file(dst_dir, filenames, src_name, fps=24.0)
    print("\nmp4 generated: {}\n".format(mp4_filename))

def _get_root_dir():
    import os 

    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    return dir_root

def _add_root_in_sys_path():
    import sys 

    dir_root = _get_root_dir()
    if dir_root not in sys.path:
        sys.path.append(dir_root)

if __name__ == '__main__':
    _add_root_in_sys_path()
    gen_mp4()


#!/bin/bash
#ffmpeg -framerate 24 -i _gen_render_little_star/20220306_%06d.png output.mp4
