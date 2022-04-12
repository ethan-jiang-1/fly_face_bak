import mediapipe as mp 
import math
import cv2
import numpy as np

_RED = (48, 48, 255)

FMV_FACE_OVAL = (127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234)
FMV_LEFT_EYE = (382, 398, 384, 385, 386, 387, 388, 466, 263, 263, 249, 390, 373, 374, 380, 381)
FMV_RIGHT_EYE = (155, 173, 157, 158, 159, 160, 161, 246, 33, 33, 7, 163, 144, 145, 153, 154)
FMV_MOUTH_INNER = (310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311)
FMV_MOUTH_OUTTER = (375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321)
FMV_LEFT_IRIS =(474, 475, 476, 477)
FMV_RIGHT_IRIS = (470, 471, 472, 469)
FMV_LEFT_EYEBROW = (300, 293, 334, 296, 336,  285, 295, 282, 283, 276)
FMV_RIGHT_EYEBROW =(70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
FMV_NOSE = (8, 417, 465,  343, 437, 420, 279, 278, 439, 438, 457, 274, 1, 44, 237, 218, 219, 48, 49, 198, 217, 114, 245, 193)


FMC_FACE_OVAL = (256, 256, 0)
FMC_LEFT_EYE = (0, 0, 32)
FMC_RIGHT_EYE = (0, 0, 32)
FMC_MOUTH_INNER = (0, 0, 64)
FMC_MOUTH_OUTTER = (0, 0, 160)
FMC_LEFT_IRIS = (0, 0, 196)
FMC_RIGHT_IRIS = (0, 0, 196)
FMC_LEFT_EYEBROW = (0, 0, 255)
FMC_RIGHT_EYEBROW = (0, 0, 255)
FMC_NOSE = (0, 0, 128)


class FmxFacePaint():
    @classmethod
    def normalized_to_pixel_coordinates(cls, normalized_x, normalized_y, image_width, image_height):
        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None, None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    @classmethod
    def draw_ploypoints(cls, image, face_landmarks, fmv_vertices, fmc_color):
        image_width, image_height = image.shape[1], image.shape[0]

        contour = []
        llm = face_landmarks.landmark
        for vt in fmv_vertices:
            px, py = cls.normalized_to_pixel_coordinates(llm[vt].x, llm[vt].y, image_width, image_height) 
            if px is not None and py is not None:
                contour.append((px, py))

        np_contour = np.array(contour)
        cv2.fillConvexPoly(image, np_contour, fmc_color) 

    @classmethod
    def process_img(cls, image, mesh_results):
        if mesh_results.multi_face_landmarks is None or len(mesh_results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 
        
        return cls.paint_face_core(image, mesh_results, paint_mode="all")

    @classmethod
    def paint_face_core(cls, image, mesh_results, paint_mode="all"):
        if mesh_results is None or mesh_results.multi_face_landmarks is None:
            print("no face_landmarks found")
            return None

        image = np.zeros(image.shape, dtype=image.dtype)
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        for face_landmarks in mesh_results.multi_face_landmarks:

            if paint_mode == "all":
                cls.draw_ploypoints(image, face_landmarks, FMV_FACE_OVAL, FMC_FACE_OVAL)
            elif paint_mode == "outline":
                cls.draw_ploypoints(image, face_landmarks, FMV_FACE_OVAL, (255, 255, 255))                

            if paint_mode == "all":
                landmark_ds = mp_drawing_styles.DrawingSpec(color=_RED, thickness=1, circle_radius=2)
                connection_ds = mp_drawing_styles.DrawingSpec(color=FMC_FACE_OVAL, thickness=1)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=landmark_ds,
                    connection_drawing_spec=connection_ds)

                cls.draw_ploypoints(image, face_landmarks, FMV_LEFT_EYE, FMC_LEFT_EYE)
                cls.draw_ploypoints(image, face_landmarks, FMV_RIGHT_EYE, FMC_RIGHT_EYE)

                cls.draw_ploypoints(image, face_landmarks, FMV_LEFT_IRIS, FMC_LEFT_IRIS)
                cls.draw_ploypoints(image, face_landmarks, FMV_RIGHT_IRIS, FMC_RIGHT_IRIS)

                cls.draw_ploypoints(image, face_landmarks, FMV_MOUTH_OUTTER, FMC_MOUTH_OUTTER)
                cls.draw_ploypoints(image, face_landmarks, FMV_MOUTH_INNER, FMC_MOUTH_INNER)

                cls.draw_ploypoints(image, face_landmarks, FMV_LEFT_EYEBROW, FMC_LEFT_EYEBROW)
                cls.draw_ploypoints(image, face_landmarks, FMV_RIGHT_EYEBROW, FMC_RIGHT_EYEBROW)
                cls.draw_ploypoints(image, face_landmarks, FMV_NOSE, FMC_NOSE)

        #print(image.shape, image.dtype)
        return image
