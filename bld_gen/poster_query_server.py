#import os
#import sys 
#import random


class PosterQueryServer(object):
    kls_output_folder = None
    kls_map_subfolders = None

    @classmethod
    def get_poster(cls,  hair_id, beard_id, face_id, gender):

        picked_url = None

        return picked_url, None

def do_exp():
    hair_id = 0
    face_id = 0
    beard_id = 0

    filename, _ = PosterQueryServer.get_poster(hair_id, face_id, beard_id, "M")
    print("post for hid:{} fid: {} bid: {} is {}".format(hair_id, face_id, beard_id, filename))
    return filename

if __name__ == '__main__':
    do_exp()
