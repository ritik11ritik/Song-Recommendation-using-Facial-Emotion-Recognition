#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:56:27 2021

@author: rg
"""

import numpy as np
import cv2
import os
from keras.models import load_model

def snapshot():
    cap = cv2.VideoCapture(0)
    name = "Snapshot.jpg"
    while True:
        ret, frame = cap.read()
        if not ret:
            print ("Can't access camera")
            break
        else:
            cv2.putText(frame,
                        "Press q to take snapshot", 
                        (50,50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        1,
                        cv2.LINE_AA)
            cv2.imshow("Video", frame)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                cv2.imwrite(name, frame)
                break
    cap.release()
    cv2.destroyAllWindows()
    emotion = emotion_from_camera(name)
    os.remove(name)
    return emotion
            
        

def emotion_from_camera(name):
    model = load_model("model.h5")
    face_cascade = cv2.CascadeClassifier()
    img = cv2.imread(name)
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
    face_cascade.load(cv2.samples.findFile("haarcascade_frontalface_default.xml"))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Haar Cascade
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        faceROI = gray[y:y+h, x:x+w]
        faceROI = cv2.resize(faceROI, (48, 48),interpolation = cv2.INTER_NEAREST)
        faceROI = np.expand_dims(faceROI, axis = 0)
        faceROI = np.expand_dims(faceROI, axis = 3)
        prediction = model.predict(faceROI)

    return labels[int(np.argmax(prediction))]
        