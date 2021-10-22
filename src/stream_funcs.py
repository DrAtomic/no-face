import cv2
import numpy as np




def capture_cam():
    """captures the webcam


    """
    cam = cv2.VideoCapture(0)
    
    while True:
        ret_val, img = cam.read()

        cv2.imshow('no face', img)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
