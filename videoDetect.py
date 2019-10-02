# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
fig = plt.figure()
plt.rcParams['figure.figsize'] = (224, 224)

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Create a VideoCapture object
cap = cv2.VideoCapture("videos/video.mp4")

i = 0
j = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist_full = cv2.calcHist([gray],[0],None,[256],[0,256])
        plt.hist(hist_full)
        j += 1
        plt.savefig("Hist/frame{}.png".format(j))
        
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x,y,w,h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            i += 1            
            cv2.imwrite("Pessoas/face{}.jpg".format(i), roi_color)
            gray1 = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            hist_full1 = cv2.calcHist([gray1],[0],None,[256],[0,256])
            plt.hist(hist_full1)            
            plt.savefig("HistFace/framaFace{}.png".format(i))
            
    else: 
        break

cap.release()
cv2.destroyAllWindows()