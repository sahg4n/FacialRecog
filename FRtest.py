import cv2 
import face_recognition as fr
import numpy as np

original_image = cv2.imread('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/22birthday.jpg')

sah = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/sg.jpg')
encSah = fr.face_encodings(sah)[0]
shamm = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/shamm1.jpg')
encShamm = fr.face_encodings(shamm)[0]
akhil = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/akhil1.jpg')
encAkhil = fr.face_encodings(akhil)[0]
niharika = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/niharika1.jpg') 
encNiharika = fr.face_encodings(niharika)[0]

shru = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/shru.jpg')
encShru = fr.face_encodings(shru)[0]
mum = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/mum.jpg')
encMum = fr.face_encodings(mum)[0]
pop = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/pop.jpg') 
encPop = fr.face_encodings(pop)[0]

known_face_encodings = [encSah, encAkhil, encShamm, encNiharika, encShru, encMum, encPop]
known_face_names = ['Sah', 'Akhil', 'Sham', 'Nih', 'Shru', 'Mum', 'Pop']
matchedFaceEnc = []


image_to_recognize = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/22birthday.jpg')
all_locations = fr.face_locations(image_to_recognize, model='hog')
all_encodings = fr.face_encodings(image_to_recognize, all_locations)
print('There are {} no. of faces in the image'.format(len(all_locations)))
for current_face_location, current_face_encoding in zip(all_locations, all_encodings):
    matchedFaceEnc.clear()
    top, right, bottom, left = current_face_location
    print('Found face at top:{}, right:{}, bottom:{}, and left:{}'.format(top, right, bottom, left))
    matches = fr.compare_faces(known_face_encodings, current_face_encoding)
    print(matches)
    name_person = 'unknown'
    if True in matches:
        faceDist = fr.face_distance(known_face_encodings, current_face_encoding)
        index = np.where(faceDist == np.amin(faceDist))
        name_person = known_face_names[index[0][0]]
        

        # faceDist = fr.face_distance(known_face_encodings, current_face_encoding)
        # index = np.where(faceDist == np.amin(faceDist))
        # name_person = known_face_names[index[0][0]]
        for i,face_distance in enumerate(faceDist):
            print("The {},{} matching percentage is {} against the sample {}".format(top, bottom, round(((1-float(face_distance))*100),2), known_face_names[i]))

    
    # for i in matches:
    #         if(matches[i]):
    #             print(known_face_names[i])

    # faceDist = fr.face_distance(known_face_encodings, current_face_encoding)
    # index = np.where(faceDist == np.amin(faceDist))
    # name_person = known_face_names[index[0][0]]
    # for i,face_distance in enumerate(faceDist):
    #     print("The {},{} matching percentage is {} against the sample {}".format(top, bottom, round(((1-float(face_distance))*100),2), known_face_names[i]))

    cv2.rectangle(original_image, (left, top), (right, bottom), (0,0,255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_person+ str(top)+''+str(bottom), (left, bottom), font, 3.9, (255,255,255))

resized = cv2.resize(original_image, (960,540))
cv2.imshow("this",resized)
cv2.waitKey()

