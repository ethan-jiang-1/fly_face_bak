
def main():
    #do_exp_face_mesh()
    do_exp_selfie()

def do_exp_face_mesh():
    from exp_mediapipe.exp_face_mesh import do_exp
    do_exp()

def do_exp_selfie():
    from exp_mediapipe.exp_selfie import do_exp
    do_exp()

if __name__ == '__main__':
    main()

