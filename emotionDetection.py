import cv2
import face_recognition as fr
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

allFaceLocs  = []
emotionsLabel = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
webcamStream = cv2.VideoCapture(0)
faceExpressionModel = model_from_json(open("/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/facial_expression_model_structure.json", "r").read())
faceExpressionModel.load_weights('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/facial_expression_model_weights.h5')

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
        cv2.rectangle(curFrame, (left, top), (right, bottom), (0,0,255), 2)
        curFaceFrame = curFrame[top:bottom, left:right]
        curFaceFrame = cv2.resize(cv2.cvtColor(curFaceFrame, cv2.COLOR_BGR2GRAY), (48,48)) #convert to grayscale and then resize to 48,48 px size
        imgPixels  = image.img_to_array(curFaceFrame)
        imgPixels = np.expand_dims(imgPixels, axis=0)
        imgPixels /= 255
        expressionPrediction = faceExpressionModel.predict(imgPixels)
        emotion = emotionsLabel[np.argmax(expressionPrediction[0])]

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(curFrame, emotion, (left, bottom), font, 0.5, (255,255,255), 1)


    cv2.imshow('webcam', curFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcamStream.release()
cv2.destroyAllWindows()