import cv2
import numpy as np


face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')

feed = cv2.VideoCapture(0)
id = raw_input('User ID: ')
idNumber = 0
while (True):
    ret, img = feed.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    for(x,y,w,h) in faces:
        idNumber += 1
        cv2.imwrite('dataSet/User.'+str(id)+'.'+str(idNumber)+'.jpg', gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,0),2)
        cv2.waitKey(100)
    cv2.imshow('Face', img)
    cv2.waitKey(1)
    if(idNumber > 20):
        break

feed.release()
cv2.destroyAllWindows()
