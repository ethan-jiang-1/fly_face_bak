#import os
#import sys 


class PosterQuery(object):
    @classmethod
    def get_poster(cls,  hair_id, face_id, beard_id):
        return None 

def do_exp():
    hair_id = 0
    face_id = 0
    beard_id = 0

    filename = PosterQuery.get_poster(hair_id, face_id, beard_id)
    print("post for hid:{} fid: {} bid: {} is {}".format(hair_id, face_id, beard_id, filename))
    return filename

if __name__ == '__main__':
    do_exp()
