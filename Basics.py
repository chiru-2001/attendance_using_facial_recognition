import cv2
import numpy
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/elon_musk_train.jpg');
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB);
imgTest = face_recognition.load_image_file('ImagesBasic/bill_gates.jpg');
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB);


faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
#print(faceLoc)
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc_test = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
#print(faceLoc)
cv2.rectangle(imgTest,(faceLoc_test[3],faceLoc_test[0]),(faceLoc_test[1],faceLoc_test[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {faceDis[0],2}',(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),2)
cv2.imshow('Elon Musk Train',imgElon);
cv2.imshow('Elon Musk Test',imgTest);
cv2.waitKey(0);