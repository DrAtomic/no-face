import cv2
import numpy as np
from detect_face import find_face
from modify_face import modify_face


def capture_cam():
    """captures the webcam


    """
    cam = cv2.VideoCapture(0)
    
    while True:
        ret_val, img = cam.read()
        faces = find_face(img)
        if faces != []:
            modify_face(img,faces)
        cv2.imshow('no face', img)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

capture_cam()
