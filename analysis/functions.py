import cv2 as cv
import numpy as np
from numpy.lib.type_check import imag
import helpers 
from PIL import Image, ImageEnhance
import logging as log
import matplotlib
import matplotlib.pyplot as plt

from skimage.filters import (threshold_niblack)
from skimage import img_as_ubyte


def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    

def median_blur(images):
    return [cv.medianBlur(img, 3) for img in images]
    
def remove_dark(medians):
    for median in medians:
        median[np.where(median < 0)] = 0
    return medians

def conservative_smoothing(images):
    smooth_imgs = []
    for img in images[:1]:
        w, h = img.shape
        log.debug("Width: {0}, Height: {1}".format(w, h))
        conservativedata = np.zeros((w, h), np.uint16)
        radius = 2
        for y in range(h):
            #for each pixel
            for x in range(w):
                minG = np.iinfo(np.uint16).max
                maxG = np.iinfo(np.uint16).min

                for i  in range(-radius, radius + 1):
                    t = y + i

                    #skip row
                    if t < 0:
                        continue
                    #break
                    if t >= h:
                        break

                    #for each kernel column
                    for j in range(-radius, radius + 1):
                        if j == 0 and i == 0:
                            continue
                        t = x + j

                        #skip column
                        if t < 0 or t >= w:
                            continue
                        #find MIN and MAX values
                        #log.debug('y: {0}, i: {1}, t: {2}'.format(y, i, t))
                        _v = img[t, y + i]

                        if _v < minG:
                            minG = _v
                        if _v > maxG:
                            maxG = _v
                #set destination pixel
                _v = img[x, y]
                _v= (_v > maxG) and maxG or ((_v < minG) and minG or _v)
                conservativedata[x, y] = _v
        smooth_imgs.append(conservativedata)
    return smooth_imgs
            


def max_pixel(images):
    max_pxl = []
    [max_pxl.append(helpers.find_max(img)) for img in images]
    return max(max_pxl)

def normalize_images(images, white):
    return [cv.normalize(img, img, 0, white, cv.NORM_MINMAX) for img in images]

def logarithm(images):
    log_images = []
    for img in images:
        # Apply log transformation method
        c = 65535 / np.log(1 + np.max(img))
        log_image = c * (np.log(img + 1))

        # Specify the data type so that
        # float value will be converted to int
        log_images.append(np.array(log_image, dtype = np.uint16))
    return log_images


def changeContrastScale(images, alpha):
    return [cv.convertScaleAbs(img, alpha=alpha) for img in images]

def convertTo8bit(images):
    return [(img/256).astype('uint8') for img in images]


#https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
#Too much Complexity in this method.
def changeContrast(images, alpha):
    constrast_img = []
    for img in images:
        new_image = np.zeros(img.shape, img.dtype)
        # Do the operation new_image(i,j) = alpha*image(i,j) + beta
        # Instead of these 'for' loops we could have used simply:
        # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        # but we wanted to show you how to access the pixels :)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                new_image[y,x] = np.clip(alpha*img[y,x], 0, 65535)
        constrast_img.append(new_image)
    return constrast_img


#https://www.geeksforgeeks.org/python-pil-imageenhance-color-and-imageenhance-contrast-method/
def changeContrastPillow(images, alpha):
    contrast = []
    for img in images:
        img = Image.fromarray(np.uint8(img))
        im3 = ImageEnhance.Contrast(img)
        contrast.append(np.asarray(im3.enhance(alpha), dtype=np.uint8))
    return contrast

def erode(images, kernel, iter):
    return [cv.erode(img, kernel=kernel, iterations=iter) for img in images]

def dilation(images, kernel, iter) :
    return [cv.dilate(img, kernel=kernel, iterations=iter) for img in images]

def niBlackThresholdScimage(images, window_size, k):
    thresh = []
    binary = []
    for img in images:
        thresh_niblack = threshold_niblack(img, window_size=window_size, k=k)
        binary_niblack = img > thresh_niblack
        thresh.append(thresh_niblack)
        binary.append(img_as_ubyte(binary_niblack))
    return thresh, binary

def findContours(images):
    cnts_all_imgs = []
    for img in images:
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts_all_imgs.append(contours)
    return cnts_all_imgs
    #return [cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) for img in images]


#https://www.programcreek.com/python/?code=fanghon%2Flpr%2Flpr-master%2Fhyperlpr_py3%2Fniblack_thresholding.py
def niBlackThreshold(  src,  blockSize,  k,  binarizationMethod= 0 ):
    thresh_imgs = []
    for img in src:
        mean = cv.boxFilter(img,cv.CV_32F,(blockSize, blockSize),borderType=cv.BORDER_REPLICATE)
        sqmean = cv.sqrBoxFilter(img, cv.CV_32F, (blockSize, blockSize), borderType = cv.BORDER_REPLICATE)
        variance = sqmean - (mean*mean)
        stddev  = np.sqrt(variance)
        thresh = mean + stddev * float(-k)
        thresh = thresh.astype(img.dtype)
        niblack = (img>(thresh+2))*255
        niblack = niblack.astype(np.uint8)
        thresh_imgs.append(niblack)
    return thresh_imgs

def quadriterals(contours_all_imgs):
    # returns a double list. For each image the value is a list of all 
    # accepted contours.
    return list(map(find_quadriterals, contours_all_imgs))
    

def find_quadriterals(cnt_per_img): 
    accepted_cnt = []
    # Searching through every region selected to 
    # find the required quadriteral.   
    #log.debug(cnt_per_img)
    for cnt in cnt_per_img:
        area = cv.contourArea(cnt)
        
        # # Shortlisting the regions based on there area.
        if area > 200: 
            approx = cv.approxPolyDP(cnt, 
                            0.04 * cv.arcLength(cnt, True), True)
   
            # Checking if the no. of sides of the selected region is 4.
            if(len(approx) == 4): 
                accepted_cnt.append(approx)
    return accepted_cnt



#<<================== Contrast Different Methods =====================>>

def BrightnessContrast(img, brightness=0):
     
    # getTrackbarPos returns the current
    # position of the specified trackbar.
    brightness = cv.getTrackbarPos('Brightness',
                                    'GEEK')
      
    contrast = cv.getTrackbarPos('Contrast',
                                  'GEEK')
  
    effect = controller(img, brightness, 
                        contrast)
  
    # The function imshow displays an image
    # in the specified window
    cv.imshow('Effect', effect)
  
def controller(img, brightness=255,
               contrast=127):
    
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
  
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
  
    if brightness != 0:
  
        if brightness > 0:
  
            shadow = brightness
  
            max = 255
  
        else:
  
            shadow = 0
            max = 255 + brightness
  
        al_pha = (max - shadow) / 255
        ga_mma = shadow
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv.addWeighted(img, al_pha, 
                              img, 0, ga_mma)
  
    else:
        cal = img
  
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv.addWeighted(cal, Alpha, 
                              cal, 0, Gamma)
  
    # putText renders the specified text string in the image.
    cv.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
    return cal
  

def test(image):
    img = image.copy()
    cv.namedWindow('GEEK')
  
    # The function imshow displays an 
    # image in the specified window.
    cv.imshow('GEEK', image)
  
    # createTrackbar(trackbarName, 
    # windowName, value, count, onChange)
     # Brightness range -255 to 255
    cv.createTrackbar('Brightness',
                       'GEEK', 255, 2 * 255,
                       BrightnessContrast) 
      
    # Contrast range -127 to 127
    cv.createTrackbar('Contrast', 'GEEK',
                       127, 2 * 127,
                       BrightnessContrast)  
  
      
    BrightnessContrast(img, 0)
  

def calculateContrast(img):
    np.seterr(divide='ignore', invalid='ignore')
    # convert to LAB color space
    lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)

    # separate channels
    L,A,B=cv.split(lab)

    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel = np.ones((5,5),np.uint8)
    min = cv.erode(L,kernel,iterations = 3)
    max = cv.dilate(L,kernel,iterations = 3)

    # convert min and max to floats
    min = min.astype(np.float64) 
    max = max.astype(np.float64) 

    # compute local contrast
    contrast = (max-min)/(max+min)

    # get average across whole image
    average_contrast = 100*np.mean(contrast)

    print(str(average_contrast)+"%")

    Y = cv.cvtColor(img, cv.COLOR_BGR2YUV)[:,:,0]

    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)

    # compute contrast
    contrast = (max-min)/(max+min)
    print(min,max,contrast)