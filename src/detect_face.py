from mtcnn import MTCNN
import numpy as np
import cv2

detector = MTCNN()

def find_face(img):
    """finds the face in an image/stream

    Args:
       img: gray image to find faces

    Returns:
        list of dictionaries of faces with keypoints

    """

    min_conf = 0.9
    pixels = np.asarray(img)
    global detector

    detected = detector.detect_faces(pixels)
    faces = [i for i in detected if i['confidence'] >= min_conf]

    return faces
