import sys
import os 

try:
    from dg_base import DgBase
except:
    from .dg_base import DgBase


class DgOutline(DgBase):
    def __init__(self):
        super(DgOutline, self).__init__()

    def aug(self, img):
        print("aug")


def do_exp():
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    dg = DgOutline()
    dg.aug(None)
    del dg 


if __name__ == '__main__':
    do_exp()
