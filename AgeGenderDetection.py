import cv2
import face_recognition as fr

webcamStream = cv2.VideoCapture(0)

allFaceLocs = []

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
        faceFrame = curFrame[top:bottom, left:right]
        
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
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
        cv2.rectangle(curFrame, (left, top), (right, bottom), (0,0,255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(curFrame, gender+" "+age+"yrs", (left,bottom+20), font, 0.5, (0,255,0),1)

    cv2.imshow('webcam', curFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcamStream.release()
cv2.destroyAllWindows()

