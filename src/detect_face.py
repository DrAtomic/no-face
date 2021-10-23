import dlib
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

cam = cv2.VideoCapture(0)

def get_region(img,point):
    """gets a region of the face of the face with the most flesh tone

    Args:
       img: croped image of face
       point: face point

    Returns:
        region of fleshy face

    """
    w = 50
    h = 50
    region = img[point[1]-h:point[1], point[0]-w:point[0]]
    return region

def get_avg(region):
    """gets the avg pixel value of fleshy region

    Args:
       region: fleshy region

    Returns:
        avg of fleshy region

    """
    average = (region.sum(axis=1).sum(axis=0)) / (region.shape[0] * region.shape[1])
    return average

def create_mask(img,point,color):
    """creates a box around a set of points

    Args:
       img: the image
       point: center point
       color: color of mask

    Returns:
        mask

    """
    zeros = np.zeros(img.shape,dtype=np.uint8)
    mask = cv2.ellipse(zeros,point,(150,125),90,0,360,color,-1)
    return mask

def edit_face(img):
    """applies the color to the face

    Args:
       img: the image

    Returns:
        faceless person

    """
    
    white_mask = create_mask(img,face_points[29],(255,255,255))
    white_region_face = cv2.bitwise_or(img,white_mask)

    flesh_region = get_region(img,face_points[30])
    flesh_avg = get_avg(flesh_region)

    img[np.all(white_region_face == (255,255,255),axis=-1)] = flesh_avg

    return img

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        #img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2)
        landmarks = predictor(gray,face)
        face_points = []
        for i in range(len(landmarks.parts())):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            face_points.append([x,y])
            # cv2.circle(img,(x,y),2,(0,0,255),2)
            # cv2.putText(img,str(i),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)

    face_points = np.array(face_points)
    img = edit_face(img)
    cv2.imshow("noface",img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
