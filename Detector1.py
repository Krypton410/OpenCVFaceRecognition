import cv2
import numpy as np


face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
feed = cv2.VideoCapture(0)
rec = cv2.cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainData.yml')




idNumber = 0
while (True):
    ret, img = feed.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    for(x,y,w,h) in faces:
        
       
        cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,0),2)
        id, conf = rec.predict(gray[y:y+h,x:x+w])
        cv2.putText(img, str(id), cv2.FONT_HERSHEY_PLAIN,(x,y+h),(0,255,0))



   
    cv2.imshow('Face', img)
    if(cv2.waitKey() == ord('q')):
        break

feed.release()
cv2.destroyAllWindows()
