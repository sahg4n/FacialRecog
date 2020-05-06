import cv2
import face_recognition as fr

webcamStream = cv2.VideoCapture(0)

allFaceLocs  = []

while True:
    ret, curFrame = webcamStream.read()
    currFrameSmall = cv2.resize(curFrame, (0,0), fx=0.25, fy=0.25)
    faceLoc = fr.face_locations(currFrameSmall, 2, 'hog')
    for index, curFaceLoc in enumerate(faceLoc):
        top, right, bottom, left = curFaceLoc
        top = top*4
        right = right*4
        bottom = bottom*4
        left = left*4
        print('Found face {} at top:{}, right:{}, bottom:{}, and left:{}'.format(index+1, top, right, bottom, left))
        frameToBlur = curFrame[top:bottom, left:right]
        blurredFrame = cv2.GaussianBlur(frameToBlur, (99,99), 9)
        curFrame[top:bottom, left:right] = blurredFrame
        cv2.rectangle(curFrame, (left, top), (right, bottom), (0,0,255), 2)
    cv2.imshow('webcam', curFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcamStream.release()
cv2.destroyAllWindows()