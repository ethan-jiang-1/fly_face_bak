import numpy as np
import math
import cv2

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)

FMB_UPPER_L2R = (147, 187, 205, 36, 203, 98, 97, 2, 326, 327, 423, 266, 425, 411, 376)
FMB_RIGHT_T2B = (401, 435, 367, 364)
FMB_BUTTOM_R2L = (394, 395, 369, 396, 175, 171, 140, 170, 169)
FMB_LEFT_B2T = (135, 138, 215, 177)

class FmxMeshBeard():
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
                contour.append([px, py])

        np_contour = np.array(contour)
        #cv2.polylines(image, [np_contour], 10, fmc_color)
        cv2.fillPoly(image, [np_contour], fmc_color) 

    @classmethod
    def process_img(cls, image, mesh_results):
        if mesh_results.multi_face_landmarks is None or len(mesh_results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 

        image_mask = cls._get_beard_mask(image, mesh_results)

        img_bread_color = cv2.bitwise_and(image, image, mask = image_mask)

        h, w = image.shape[:2]
        mask = np.zeros([h+2, w+2],np.uint8)
        #cv2.floodFill(img_bread, mask, (220, 250), (0, 255, 255), (100, 100, 100), (50, 50 ,50), cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(img_bread_color, mask, (220, 250), (216, 216, 216), (100, 100, 100), (50, 50 ,50), cv2.FLOODFILL_FIXED_RANGE)

        img_bread_gray = cv2.cvtColor(img_bread_color, cv2.COLOR_BGR2GRAY)

        _, th1 = cv2.threshold(img_bread_gray, 100, 255, cv2.THRESH_BINARY)

        #img_blur = cv2.GaussianBlur(img_bread_gray, (5, 5), 0)
        #_, th3 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_bread = th1

        print(img_bread.shape, img_bread.dtype)
        return img_bread

    @classmethod
    def _get_beard_mask(cls, image, mesh_results):
        image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int8)
        print("len(mesh_results.multi_face_landmarks)", len(mesh_results.multi_face_landmarks))
        for face_landmarks in mesh_results.multi_face_landmarks:
            pts = []
            pts.extend(FMB_UPPER_L2R)
            pts.extend(FMB_RIGHT_T2B)
            pts.extend(FMB_BUTTOM_R2L)
            pts.extend(FMB_LEFT_B2T)

            cls.draw_ploypoints(image_mask, face_landmarks, pts, (255))
        return image_mask
