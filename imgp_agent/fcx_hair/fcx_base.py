import math 
import cv2
import numpy as np

FMB_PX_ALTER = {"U": (0, 0), "R":(0.00, 0), "L":(-0.00, 0), "B":(0, 0.00)}
FMB_FILL_COLOR = (216, 216, 216)

class FcxBase():
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
            num = vt
            rx, ry = llm[num].x, llm[num].y,
            px, py = cls.normalized_to_pixel_coordinates(rx, ry,  image_width, image_height) 
            if px is not None and py is not None:
                contour.append([px, py])

        np_contour = np.array(contour)
        #cv2.polylines(image, [np_contour], 10, fmc_color)
        cv2.fillPoly(image, [np_contour], fmc_color) 
    
    @classmethod
    def draw_ploypoints_alter(cls, image, face_landmarks, fmv_vertices, fmc_color):
        image_width, image_height = image.shape[1], image.shape[0]

        contour = []
        llm = face_landmarks.landmark
        for vt in fmv_vertices:
            num = vt
            alter_direction = None 
            if hasattr(vt, "__len__"):
                num = vt[0]
                alter_direction = vt[1]

            rx, ry = llm[num].x, llm[num].y,
            if alter_direction is not None: 
                if alter_direction in FMB_PX_ALTER:
                    dx, dy = FMB_PX_ALTER[alter_direction]
                    rx += dx 
                    ry += dy
                else:
                    print("unknown direction", alter_direction)

            px, py = cls.normalized_to_pixel_coordinates(rx, ry,  image_width, image_height) 
            if px is not None and py is not None:
                contour.append([px, py])

        np_contour = np.array(contour)
        #cv2.polylines(image, [np_contour], 10, fmc_color)
        cv2.fillPoly(image, [np_contour], fmc_color) 

    @classmethod
    def get_vt_coord(cls, image, face_landmarks, vt):
        image_width, image_height = image.shape[1], image.shape[0]
        llm = face_landmarks.landmark 
        px, py = cls.normalized_to_pixel_coordinates(llm[vt].x, llm[vt].y, image_width, image_height) 
        if px is not None:
            return px, py
        return None      

    @classmethod
    def flood_fill(cls, image, pt_seed=(0, 0), color_fill=FMB_FILL_COLOR):
        h, w = image.shape[:2]
        mask_flood = np.zeros([h+2, w+2],np.uint8)
        cv2.floodFill(image, mask_flood, pt_seed, color_fill, flags=cv2.FLOODFILL_FIXED_RANGE)
