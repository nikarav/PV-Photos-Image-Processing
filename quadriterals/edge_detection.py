import cv2
import os 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola,try_all_threshold, threshold_local)
from PIL import Image, ImageEnhance
image_folder = r"C:\Users\morte\Desktop\Test_4_1\pre-processed"

def estimate_polygons(contours):
    # estimate polygons based on contours
    return list(map(find_corners, contours))
def find_corners(contour, precision = 0.04):
    epsilon = precision * cv.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)
    return corners

def keep_quadrilaterals(polygons):
    # filter out polygons not a quadrilateral
    return list(filter(contour_is_square, polygons))

def flip_square(s):
    '''
    flips a quad so the left upper corner is first in the array, and then the
    points are counter clockwise from that.

    '''
    index = np.argmin(np.sum(s[:,0,:],axis=1))
    s = np.roll(s, 4-index, axis = 0)
    return s

def contour_is_square(contour):
    if (len(contour) == 4) and (cv.contourArea(contour) > 2000 ):
        # it needs to have some size since it also finds very tiny ones, which
        # cant be a cell
        contour = flip_square(contour)
        topleft, bottomleft, bottomright, topright = contour[:,0,:]
        if not np.array_equal(contour, cv.convexHull(contour)):
            #checking if contour is a rectangle but not the best way for sure
            return False
        return True
    else:
        return False

def log_transform(image, integer=16):
    max_pixel = np.max(image)
    c = 2**integer / (np.log(1 + max_pixel))
    log_transform = c * np.log(image + 1)
    return log_transform.astype(f'uint{integer}')


def adjust_contrast(img, factor):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    contrast_increase = enhancer.enhance(factor)
    return np.array(contrast_increase)

    
def detect_quadrilaterals(image, alpha = 1.3, opening_iterations=2, kernel_size=3, window_size=11,
                          k = 11, offset = 15, integer=8):
    log = log_transform(image, integer=integer)
    alpha = alpha # Contrast control (1.0-3.0)
    img_contrast = adjust_contrast(log, alpha)
    iterations = opening_iterations
    kernel = np.ones((kernel_size, kernel_size))
    img_erode = cv.erode(img_contrast, kernel, iterations = iterations)
    img_dilate = cv.dilate(img_erode, kernel, iterations = iterations)
    thresh_niblack =  threshold_niblack(img_dilate, window_size=window_size, k=k)
    binary_niblack = image > (thresh_niblack+offset) # which one should it be??
    # binary_niblack = image > (thresh_niblack+offset)
    thresholded_image = (1 * binary_niblack).astype('uint8') # needs to be uint for contours method
    contours = cv.findContours(thresholded_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    contours_polygons = estimate_polygons(contours)
    quadrilaterals = keep_quadrilaterals(contours_polygons)
    return quadrilaterals

if __name__ == "__main__":
    
    for image_name in os.listdir(image_folder):
        
        if ".tif" not in image_name:
            continue
        path = os.path.join(image_folder, image_name)
        img_gray = cv.imread(path, -1)
        img_gray8 = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)
        img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR) 
        log = log_transform(img_gray8, integer=8)
        
        # cv.imshow('log', log)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        alpha = 1.3# Contrast control (1.0-3.0)
        img_contrast = adjust_contrast(log, alpha)
        # cv.imshow('img_contrast', img_contrast)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        kernel = np.ones((3,3))
        iterations = 2
        
        img_erode = cv.erode(img_contrast, kernel, iterations = iterations)
        img_dilate = cv.dilate(img_erode, kernel, iterations = iterations)
        
    
        # cv.imshow('img_opened', img_dilate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        thresh_niblack =  threshold_niblack(img_dilate, window_size=11, k=11)
        offset = 15
        binary_niblack = img_gray8 > (thresh_niblack + offset)
        thresholded_image = (1 * binary_niblack).astype('uint8') # needs to be uint for contours method
        plt.imshow(thresholded_image, cmap='gray')
        plt.show()
        cv.imshow('threshold', thresholded_image*img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
        contours = cv.findContours(thresholded_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
        contours_polygons = estimate_polygons(contours)
        quadrilaterals = keep_quadrilaterals(contours_polygons)
        cv.drawContours(img, quadrilaterals, -1, (255,255,0),3) 
        cv.imshow('ger', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break