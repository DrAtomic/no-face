import cv2

def modify_face(img,face_coordinates):
    """this modifies the face

    Args:
       img: the gray image with the face in it
       face_coordinates: the face coordinates

    """
    
    bounding_box = face_coordinates[0]['box']
    crop = img[bounding_box[1]-50:bounding_box[1]+bounding_box[3]+50,
               bounding_box[0]-50:bounding_box[0]+bounding_box[2]+50]

    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    
    edges_blur = cv2.GaussianBlur(gray, (5,5),0)
    contours_blur = cv2.GaussianBlur(gray, (23,23),0)
    
    detected_edges = cv2.Canny(edges_blur,33,33*3,5)
    thresh = cv2.threshold(contours_blur, 127, 225, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #get the largest contour that isnt the big one

    cv2.drawContours(crop,cnts,-1,(0,255,0),3)
    
    return detected_edges, thresh
