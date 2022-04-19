# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:37:52 2021

@author: morte
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils 
import os

class Stitcher:
    def __init__(self):
        self.stitcher = cv2.Stitcher_create(1)
        self.images = []
        self.stitch = 0
        
    def load_images(self, folder):
        counter =0
        for image in os.listdir(folder):
            if ".tif" in image:
                
                print("Loaded image", image)
                path = os.path.join(folder, image)
                loaded_image = cv2.imread(path)
                loaded_image *= 5
                self.images.append(loaded_image)
                counter +=1
                if counter>2:
                    break
            
    def stitch_images(self):
        status, stitch = self.stitcher.stitch(self.images)
        self.stitch = stitch
        
    def fill_to_edges(self):
        
        stitched = cv2.copyMakeBorder(self.stitch, 10, 10, 10, 10,
        			cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        			cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # allocate memory for the mask which will contain the
        # rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        minRect = mask.copy()
        sub = mask.copy()
        		# keep looping until there are no non-zero pixels left in the
        		# subtracted image
        while cv2.countNonZero(sub) > 0:
        			# erode the minimum rectangular mask and then subtract
        			# the thresholded image from the minimum rectangular mask
        			# so we can count if there are any non-zero pixels left
        	minRect = cv2.erode(minRect, None)
        	sub = cv2.subtract(minRect, thresh)
            
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        			cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        # use the bounding box coordinates to extract the our final
        # stitched image
        stitched = stitched[y:y + h, x:x + w]
        self.stitch = stitched
    
    def show(self):
        cv2.imshow("image", self.stitch)
        cv2.waitKey(0) 