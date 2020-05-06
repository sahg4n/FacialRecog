import cv2
import face_recognition as fr
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json


imageToDetect = cv2.imread('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/g.jpg')
emotionsLabel = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
faceExpressionModel = model_from_json(open("/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/facial_expression_model_structure.json", "r").read())
faceExpressionModel.load_weights('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/facial_expression_model_weights.h5')


#cv2.imshow("1",imageToDetect)

allFaces = fr.face_locations(imageToDetect, 1, model='hog')


for index, curFaceLoc in enumerate(allFaces):
    top, right, bottom, left = curFaceLoc
    #print('Found face {} at top:{}, right:{}, bottom:{}, and left:{}'.format(index+1, top, right, bottom, left))
    currentFace = imageToDetect[top:bottom, left:right]
    cv2.rectangle(imageToDetect, (left, top), (right, bottom), (0,0,255), 2)
    curFaceFrame = imageToDetect[top:bottom, left:right]
    curFaceFrame = cv2.resize(cv2.cvtColor(curFaceFrame, cv2.COLOR_BGR2GRAY), (48,48)) #convert to grayscale and then resize to 48,48 px size
    imgPixels  = image.img_to_array(curFaceFrame)
    imgPixels = np.expand_dims(imgPixels, axis=0)
    imgPixels /= 255
    expressionPrediction = faceExpressionModel.predict(imgPixels)
    emotion = emotionsLabel[np.argmax(expressionPrediction[0])]

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(imageToDetect, emotion, (left, bottom), font, 2.9, (255,255,255), 1)
resized = cv2.resize(imageToDetect, (960,540))
cv2.imshow("this",resized)
cv2.waitKey()