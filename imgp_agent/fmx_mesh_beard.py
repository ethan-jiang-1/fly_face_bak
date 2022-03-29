#import mediapipe as mp 

class FmxMeshBeard():
    @classmethod
    def process_img(cls, image, mesh_results):
        if mesh_results.multi_face_landmarks is None or len(mesh_results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 
        return None
