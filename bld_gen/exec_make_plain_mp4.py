#!/usr/bin/env python

import os
import sys 

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from bld_gen.tools.mp4_maker import Mp4Maker


IMGS_FOLDER = "_gen_render_mcd_beard.v0"
#IMGS_FOLDER = "_gen_render_mcd_beard"


def gen_mp4():
    src_dir = IMGS_FOLDER
    dst_dir = "_gen_render_mp4"

    src_dir = dir_root + os.sep + src_dir
    dst_dir = dir_root + os.sep + dst_dir

    src_name = os.path.basename(src_dir)

    filenames = Mp4Maker.get_sorted_img_files(src_dir)
    mp4_filename = Mp4Maker.create_mp4_file(dst_dir, filenames, src_name, fps=24.0)
    print("\nmp4 generated: {}\n".format(mp4_filename))

if __name__ == '__main__':
    gen_mp4()


#!/bin/bash
#ffmpeg -framerate 24 -i _gen_render_little_star/20220306_%06d.png output.mp4
