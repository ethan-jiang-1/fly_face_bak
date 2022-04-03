#!/usr/bin/env python
from bld_gen.tools_mp4.mp4_maker import Mp4Maker


IMGS_FOLDER = "_gen_render_mcd_beard"


def gen_mp4():
    src_dir = IMGS_FOLDER
    dst_dir = "_gen_render_mp4"

    filenames = Mp4Maker.get_sorted_img_files(src_dir)
    mp4_filename = Mp4Maker.create_mp4_file(dst_dir, filenames, src_dir, fps=24.0)
    print("\nmp4 generated: {}\n".format(mp4_filename))

if __name__ == '__main__':
    gen_mp4()


#!/bin/bash
#ffmpeg -framerate 24 -i _gen_render_little_star/20220306_%06d.png output.mp4
