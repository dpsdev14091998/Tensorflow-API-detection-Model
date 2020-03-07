# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:21:02 2018

@author: DEV
"""
import numpy as np
import cv2 as cv
cap = cv.VideoCapture('View_001_S2_L1.mp4')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()


# Object Detection and Tracking

from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path , "View_001_S2_L1.mp4"), output_file_path=os.path.join(execution_path , "brandnew.mp4"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    

