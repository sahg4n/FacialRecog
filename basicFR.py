import cv2
import face_recognition as fr


imageToDetect = cv2.imread('/home/sahgan/Desktop/g.jpg')

#cv2.imshow("1",imageToDetect)

allFaces = fr.face_locations(imageToDetect, 1, model='hog')


for index, curFaceLoc in enumerate(allFaces):
    top, right, bottom, left = curFaceLoc
    print('Found face {} at top:{}, right:{}, bottom:{}, and left:{}'.format(index+1, top, right, bottom, left))
    currentFace = imageToDetect[top:bottom, left:right]
    cv2.imshow('face no: '+str(index+1),currentFace)
    cv2.waitKey()