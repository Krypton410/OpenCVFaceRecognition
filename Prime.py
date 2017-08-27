import cv2
import numpy as np


face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')

feed = cv2.VideoCapture(0)

while (True):
    ret, img = feed.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,0),2)
    cv2.imshow('Face', img)
    if(cv2.waitKey(1) == ord('q')):
        break

feed.release()
cv2.destroyAllWindows()
