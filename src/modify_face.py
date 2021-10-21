import cv2

def modify_face(img,face_coordinates):
    """this modifies the face

    Args:
       img: the gray image with the face in it
       face_coordinates: the face coordinates


    """
    bounding_box = face_coordinates[0]['box']
