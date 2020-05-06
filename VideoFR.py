import cv2
import face_recognition as fr

webcamStream = cv2.VideoCapture(0)
sah = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/sg.jpg')
encSah = fr.face_encodings(sah)[0]
shamm = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/shamm1.jpg')
encShamm = fr.face_encodings(shamm)[0]
akhil = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/akhil1.jpg')
encAkhil = fr.face_encodings(akhil)[0]

knownFaceEncodings = [encSah, encShamm, encAkhil]
knownFaceNames = ["Sahh", "Shamm", "Akhil"]

allFaceLocs = []
allFaceEncodings = []
allFaceNames = []


while True:
    ret, curFrame = webcamStream.read()
    currFrameSmall = cv2.resize(curFrame, (0,0), fx=0.25, fy=0.25)
    allfaceLoc = fr.face_locations(currFrameSmall, 2, 'hog')
    allFaceEncodings = fr.face_encodings(currFrameSmall, allfaceLoc)

    for current_face_location, current_face_encoding in zip(allfaceLoc, allFaceEncodings):
        top, right, bottom, left = current_face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        matches = fr.compare_faces(knownFaceEncodings, current_face_encoding)
        name_person = 'unknown'
        if True in matches:
            firstMatchIndex = matches.index(True)
            name_person = knownFaceNames[firstMatchIndex]

        cv2.rectangle(curFrame, (left, top), (right, bottom), (0,0,255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(curFrame, name_person, (left, bottom), font, 0.6, (255,255,255), 1)

    
    cv2.imshow('webcam', curFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcamStream.release()
cv2.destroyAllWindows()

