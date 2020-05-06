import face_recognition as fr


fam = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/22birthday.jpg')
encSah = fr.face_encodings(fam)[0]
shru = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/bqmd.jpg')
encShamm = fr.face_encodings(shru)[0]
mum = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/ccfam.jpg')
encAkhil = fr.face_encodings(mum)[0]
pop = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/diwalimd.jpg') 
encNiharika = fr.face_encodings(pop)[0]
pop = fr.load_image_file('/home/sahgan/Desktop/quarantineStuff/Udemy/ComputerVisionFacialRecognition/images/tirupatishave.jpg') 
encNiharika = fr.face_encodings(pop)[0]

print('Success')