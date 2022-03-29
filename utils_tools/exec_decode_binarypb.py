import os

def main(binarypb_filename):
    from utils_inspect.inspect_solution import decode_binarypb

    if not os.path.isfile(binarypb_filename):
        raise ValueError("{} not exist".format(binarypb_filename))

    dst_filename = binarypb_filename.replace(".binarypb", ".pbtxt")

    binarypb_content = ""
    with open(binarypb_filename, "rb") as fin:
        binarypb_content = fin.read()

    decodedpb_content = decode_binarypb(binarypb_content)

    if not os.path.isfile(dst_filename):
        with open(dst_filename, "wt+") as fout:
            fout.write(decodedpb_content)
        print("gen {}".format(dst_filename))
    else:
        print(decodedpb_content)


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
    dir_parent = _get_root_dir()

    binarypb_filename = "{}/_exp_tflite_mediapipe/ref_graph_exist/selfie_segmentation_cpu.binarypb".format(dir_parent)
    main(binarypb_filename)
