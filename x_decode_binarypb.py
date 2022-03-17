from utils_inspect.inspect_solution import decode_binarypb
import os

def main(binarypb_filename):
    if not os.path.isfile(binarypb_filename):
        raise ValueError("{} not exist".format(binarypb_filename))

    dst_filename = binarypb_filename.replace(".binarypb", ".pbtxt")
    if os.path.isfile(dst_filename):
        raise ValueError("{} existed already".format(dst_filename))

    binarypb_content = ""
    with open(binarypb_filename, "rb") as fin:
        binarypb_content = fin.read()

    decodedpb_content = decode_binarypb(binarypb_content)

    with open(dst_filename, "wt+") as fout:
        fout.write(decodedpb_content)

    print("gen {}".format(dst_filename))


if __name__ == '__main__':
    binarypb_filename = "tflite_models/selfie_segmentation_cpu.binarypb"
    main(binarypb_filename)
