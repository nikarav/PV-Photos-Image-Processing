import cv2
import os 
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import random as rng

# put in your path to github folder
to_github = r"C:\Users\morte\Documents\GitHub"
image_folder = to_github + "\ImageProcessing\Data\Test_2_1\Pos0"
dark_folder = to_github + "\ImageProcessing\Data\Test_2_dark\Pos0"

def plot_image(image, title, cmap = 'gray'):
    plt.figure(dpi=300)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

image_name1 = "img_000000000_Default_000.tif"
image_name2 = "img_000000005_Default_000.tif"
path1 = os.path.join(image_folder, image_name1)
path2 = os.path.join(image_folder, image_name2)

img1 = cv.imread(path1)
img2 = cv.imread(path2)

# equalize hist to enhance brightness in order to better find contours
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img1_equ = cv.equalizeHist(img1_gray)

img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img2_equ = cv.equalizeHist(img2_gray)

# plot_image(img1_gray, "Image 1")
# plot_image(img1_equ, "Image 1 equalized")
# plot_image(img2_gray, "Image 2")
# plot_image(img2_equ, "Image 2 equalized")

"""
a loop like this will show how the thresholded binary image looks like
for i in range(100):
    plt.imshow(cv.threshold(img1_equ, i, 255, 0)[1], cmap='gray')
    plt.title(i)
    plt.show()"""

""" canny work too then the loop looks like this
for i in range(100):
    plt.imshow(cv.Canny(img1_equ, i, i * 2), cmap='gray') 
    plt.title(i)
    plt.show()"""
# canny_threshold = 50
# canny_output = cv.Canny(img1_equ, canny_threshold, canny_threshold * 2) 
# canny_output = cv.Canny(img1_equ,60,150)
# contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

def find_biggest_contour(contours):
    return max(contours, key = cv2.contourArea)

def estimate_polygon(contour, precision = 0.04):
    epsilon = precision * cv.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True) #estimates a polygon on the biggest contour
    return corners

def find_biggest_square_quadrilateral(contours):
    # contours sorted by size
    counter = 0
    n = len(contours)
    while counter < n:
        c = contours[counter]
        e = estimate_polygon(c)
        if len(e) == 4:
            return e
        counter += 1
    print("No quadrilateral found")
    return None

def find_contours(image, threshold=2):

    ret, thresh = cv.threshold(image, threshold, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # I guess that the biggest contour is the panel
    # contours = cv.findContours(img1_equ, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return contours

def draw_quadrilateral(image):
    contours = find_contours(image, threshold=2)
    corners = find_biggest_square_quadrilateral(contours)
    cv.polylines(image, [corners], True, (0,0,255), 2)
    equalized = cv.equalizeHist(image)
    cv.imshow("image" , equalized)
    cv.waitKey(0)

def flip_square(s):
    index = np.argmin(np.sum(s,axis=1))
    s = np.roll(s, 4-index, axis = 0)
    return s
    
for image in os.listdir(image_folder):
    path = os.path.join(image_folder, image)
    img = cv.imread(path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    draw_quadrilateral(img_gray)
    
    contours = find_contours(img_gray)
    corners = find_biggest_square_quadrilateral(contours)
    
    x,y,w,h = cv2.boundingRect(corners)
    #dst = np.array([[x+w,y],[x,y],[x,y+h],[x+w,y+h]])
    dst = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
    s = corners.astype(np.float32)[:,0,:]
    s = flip_square(s)
    
    dst = dst.astype(np.float32)
    M = cv.getPerspectiveTransform(s, dst)
    warp = cv.warpPerspective(img_gray, M, (600,600))
    
    warp = warp[y:y+h, x:x+w]
    
    #cv2.rectangle(warp,(x,y),(x+w,y+h),(255,0,0),1)
    
    cv.imshow("image", warp*5)
    cv.waitKey(0)
contours = find_contours(img1_gray)
corners = find_biggest_square_quadrilateral(contours)
cv.polylines(img1, [corners], True, (0,255,0), 2)

# https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
x,y,w,h = cv2.boundingRect(corners)
#dst = np.array([[x+w,y],[x,y],[x,y+h],[x+w,y+h]])
dst = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
s = corners.astype(np.float32)[:,0,:]



s = flip_square(s)

dst = dst.astype(np.float32)
M = cv.getPerspectiveTransform(s, dst)
warp = cv.warpPerspective(img1_gray,M,(600,600))

warp = warp[y:y+h, x:x+w]

#cv2.rectangle(warp,(x,y),(x+w,y+h),(255,0,0),1)

cv.imshow("image", warp*5)
cv.waitKey(0)
# consider adaptive thresholding:
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#adaptive-thresholding



# cv.drawContours(img1, contours, -1, (255,0,0),3) 
# cv2.polylines(img1, [corners], True, (0,0,255), 2)
# the polygon to its edges see doc
"""
cv2.imshow('Contours', img1*5)
cv.imshow('ger', img1_gray*5)
cv2.waitKey(0)
cv2.destroyAllWindows()


threshold = 2
ret, thresh = cv.threshold(img2_equ, threshold, 255, 0) # doesnt quite work
ret, thresh = cv.threshold(img2_gray, threshold, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# contours = cv.findContours(img2_equ, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
# cv.drawContours(img2, contours, -1, (255,0,0),3) 
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
corners = find_biggest_square_quadrilateral(contours)
#cv.drawContours(img2, contours, -1, (255,0,0),3) 
cv2.polylines(img2, [corners], True, (0,0,255), 2)

# biggest_contour = find_biggest_contour(contours)

# corners = estimate_quadrilateral(biggest_contour)
# cv2.polylines(img2, [corners], True, (0,255,0), 2)
cv2.imshow('Contours', img2*5)

cv2.waitKey(0)
cv2.destroyAllWindows()"""