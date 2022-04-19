# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:33:45 2021

@author: morte
"""
# https://colab.research.google.com/drive/11Md7HWh2ZV6_g3iCYSUw76VNr4HzxcX5?source=post_page---------------------------#scrollTo=IgBHhSxsE9go
import cv2
import os 
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import imutils

folder = r"C:\Users\morte\Documents\GitHub\ImageProcessing\Data\Test_2_1\Pos0"

image_name1 = "img_000000000_Default_000.tif"
image_name2 = "img_000000005_Default_000.tif"
path1 = os.path.join(folder, image_name1)
path2 = os.path.join(folder, image_name2)
# img2 = cv2.imread(path1)*5
# img1 = cv2.imread(path2)*5

# cv2.imshow("image", img2)
# cv2.waitKey(0)

feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'bf' # bf or knn

def threshold_gray(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)
    
    # crop the image to the bbox coordinates
    img = img[y:y + h, x:x + w]
    return img



def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)



def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None

folder = r"C:\Users\morte\Documents\GitHub\ImageProcessing\Data\Test_2_1\Pos0"

# image_name1 = "img_000000000_Default_000.tif"
# image_name2 = "img_000000005_Default_000.tif"
# path1 = os.path.join(folder, image_name1)
# path2 = os.path.join(folder, image_name2)
# img2 = cv2.imread(path1)*5
# img1 = cv2.imread(path2)*5

# cv2.imshow("image", img2)
# cv2.waitKey(0)

feature_extractor = 'brisk' # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'bf' # bf or knn    

# kpsA, featuresA = detectAndDescribe(img1, method=feature_extractor)
# kpsB, featuresB = detectAndDescribe(img2, method=feature_extractor)

# display the keypoints and features detected on both images
# fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
# ax1.imshow(cv2.drawKeypoints(img1,kpsA,None,color=(0,255,0)))
# ax1.set_xlabel("(a)", fontsize=14)
# ax2.imshow(cv2.drawKeypoints(img2,kpsB,None,color=(0,255,0)))
# ax2.set_xlabel("(b)", fontsize=14)
# plt.show()

def get_matches(feature_matching, featuresA, featuresB, kpsA, kpsB):
    plt.figure(figsize=(20,8))
    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(img1,kpsA,img2,kpsB,matches[:100],
                                None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
        img3 = cv2.drawMatches(img1,kpsA,img2,kpsB,np.random.choice(matches,100),
                                None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
    return matches

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H
    from: https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht@H, (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result
#folder = r"C:\Users\morte\Documents\GitHub\ImageProcessing\Data\Test_2_1\Pos0"

image_name1 = "img_000000000_Default_000.tif"
image_name2 = "img_000000005_Default_000.tif"
path1 = os.path.join(folder, image_name1)
path2 = os.path.join(folder, image_name2)
img1 = cv2.imread(path1)*5
img2 = cv2.imread(path2)*5
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
# cv2.imshow("image", img2)
# cv2.waitKey(0)

feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'bf' # bf or knn    


img1 = threshold_gray(img1)
img2 = threshold_gray(img2)
kpsA, featuresA = detectAndDescribe(img1, method=feature_extractor)
kpsB, featuresB = detectAndDescribe(img2, method=feature_extractor)

matches = get_matches(feature_matching, featuresA, featuresB, kpsA, kpsB)

M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
if M is None:
    print("Error!")
matches, H, status = M
"""xoffset=200
yoffset=200
offset = np.array([[1,0,xoffset],
                   [0,1,yoffset],
                   [0,0,1]])


width = img1.shape[1] + img2.shape[1]+200
height = img1.shape[0] + img2.shape[0]+200

result = cv2.warpPerspective(img1, H@offset, (width, height))
result[200-1:img2.shape[0]+200-1, 200-1:img2.shape[1]+200-1] = img2"""
result = warpTwoImages(img2, img1, H)
plt.figure(dpi=300)
plt.imshow(result)
plt.show()  

plt.figure(dpi=300)
plt.imshow(threshold_gray(result))
plt.show()   
"""
first_time = True
base = None
MINIMUM_MATCHES = 170
IMAGE_CONSTANT = 5
import time
print("ALGO START")
for image in os.listdir(folder):
    if '.tif' not in image:
        continue
    if first_time:
        path = os.path.join(folder, image)
        print(path)
        base = cv2.imread(path) * IMAGE_CONSTANT
        base = threshold_gray(base)
        first_time=False
        continue
    next_image_path = os.path.join(folder, image)
    next_image = cv2.imread(next_image_path) * IMAGE_CONSTANT
    next_image = threshold_gray(next_image)
    img2 = base
    img1 = next_image
        
    kpsA, featuresA = detectAndDescribe(img1, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(img2, method=feature_extractor)

# display the keypoints and features detected on both images
# fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
# ax1.imshow(cv2.drawKeypoints(img1,kpsA,None,color=(0,255,0)))
# ax1.set_xlabel("(a)", fontsize=14)
# ax2.imshow(cv2.drawKeypoints(img2,kpsB,None,color=(0,255,0)))
# ax2.set_xlabel("(b)", fontsize=14)
# plt.show()

    matches = get_matches(feature_matching, featuresA, featuresB, kpsA, kpsB)

    if len(matches) < MINIMUM_MATCHES:
        continue
    plt.figure(figsize=(20,10))
    plt.imshow(img1)
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(20,10))
    plt.imshow(img2)
    plt.axis('off')
    plt.show()

    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    second_time = False
    if M is None:
        print("Error!")
    (matches, H, status) = M
    print(H)

    # Apply panorama correction
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]
    
    result = cv2.warpPerspective(img1, H, (width, height))
    
    result[0:img2.shape[0], 0:img2.shape[1]] = img2[:,:]
    plt.figure(figsize=(20,10))
    plt.imshow(result)
    
    plt.axis('off')
    plt.show()

    result = threshold_gray(result)
    base = result
    #time.sleep(2)
    

# show the cropped image
plt.figure(figsize=(20,10))
plt.imshow(result)
plt.show()
"""