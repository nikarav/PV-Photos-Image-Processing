# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:42:33 2021

@author: morte

# https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980
# https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img_gray, cmap = 'gray')


ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(thresh, cmap = 'gray')

cv2.Sticher
"""

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=0,
	help="whether to crop out largest rectangular region")
args = vars(ap.parse_args())
# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []
# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)
# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)