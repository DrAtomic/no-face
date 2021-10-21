import cv2

def modify_face(img,face_coordinates):
    """this modifies the face

    Args:
       img: the image with the face in it
       face_coordinates: the face coordinates


    """
    bounding_box = face_coordinates[0]['box']
    cv2.rectangle(img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,255,0),
                  2)
