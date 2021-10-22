import dlib
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

def create_mask(img,points):
    """creates a box around a set of points

    Args:
       img: the image
       points: the points for the img

    Returns:
        idk

    """
    zeros = np.zeros(img.shape,dtype=np.uint8)
    mask = cv2.fillPoly(zeros,[points],(255,255,255))
    return mask

def top_mask(img,forhead):
    """this makes a mask from the forhead up

    Args:
       img:       img to mask
       forhead:   forhead point

    Returns:
        mask of the top half and bottom half

    """
    zeros = np.zeros(img.shape,dtype=np.uint8)
    forhead_mask = cv2.circle(zeros, forhead, 125,(255,255,255),-1)
    
    return forhead_mask

def get_region(img,points):
    """gets a region of the face of the face with the most flesh tone

    Args:
       img: croped image of face
       points: face points

    Returns:
        region of fleshy face

    """

    # get a bounding box from 41 to 1
    #                          1 to 6
    #                          6 to 49
    #                          49 to 41
    pass
    

def edit_face(img,mask):
    """applies the color to the face

    Args:
       img: the image
       mask: mask

    Returns:
        faceless person

    """
    result = cv2.bitwise_or(img,mask)
    return result

img = cv2.imread("../data/2021-10-20-224934.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray)
for face in faces:
    x1,y1 = face.left(),face.top()
    x2,y2 = face.right(),face.bottom()
    #img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2)
    landmarks = predictor(gray,face)
    face_points = []
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        face_points.append([x,y])
        cv2.circle(img,(x,y),2,(0,0,255),2)
        cv2.putText(img,str(i),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)


    face_points = np.array(face_points)
    regions = []
    regions.append(create_mask(img,face_points[0:17]))
    regions.append(top_mask(img,face_points[28]))

    
    
    cv2.imshow("test1",img)
    img = edit_face(img,regions[0])
    img = edit_face(img,regions[1])
    


    
cv2.imshow("test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
