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
AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


for index, curFaceLoc in enumerate(allFaces):
    top, right, bottom, left = curFaceLoc
    faceFrame = imageToDetect[top:bottom, left:right]    
    current_face_image_blob = cv2.dnn.blobFromImage(faceFrame, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
    gender_label_list = ['Male', 'Female']
    gender_protext = "/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/gender_deploy.prototxt"
    gender_caffemodel = "/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/gender_net.caffemodel"
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    gender_predictions = gender_cov_net.forward()
    gender = gender_label_list[gender_predictions[0].argmax()]
    age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age_protext = "/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/age_deploy.prototxt"
    age_caffemodel = "/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/solutionFIles/dataset/age_net.caffemodel"
    age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
    age_cov_net.setInput(current_face_image_blob)
    age_predictions = age_cov_net.forward()
    age = age_label_list[age_predictions[0].argmax()]
    cv2.rectangle(imageToDetect, (left, top), (right, bottom), (0,0,255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(imageToDetect, gender+" "+age+"yrs", (left,bottom+20), font, 2.9, (0,255,0),1)

resized = cv2.resize(imageToDetect, (960,540))
cv2.imshow("this",resized)
cv2.waitKey()