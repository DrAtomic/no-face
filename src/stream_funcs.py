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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = find_face(gray)

        if faces != []:
            modify_face(gray,faces)
        cv2.imshow('no face', img)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

capture_cam()
