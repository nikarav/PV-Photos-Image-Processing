import os
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import helpers
import functions as func
import argparse
import logging as log


def run(argv=None):
    parser = argparse.ArgumentParser(prog='Identify Solar Panel Cells', 
      description='')

    parser.add_argument(
        '--path',
        dest='dir',
        action='store',
        default=False,
        help='Path to folder',
        type=str,
        required=True
    )

    known_args, run_args = parser.parse_known_args(argv) 
    image_folder = known_args.dir
    process(image_folder)
  

def process(image_folder):
    # Read all images from the given folder
    images = helpers.load_images_from_folder(image_folder)

    img1 = images[0]
    img2 = images[1]
    #cv.imshow("Original 1", img1 * 5)
    
    # Step 1: Apply median smoothing blur.
    
    log.debug('Applying median smoothing blur')
    medians = func.conservative_smoothing(images)

    cv.imshow("Median 1", medians[0]*5) 

    # Step 2: Set all pixels <0 to 0 (<0 is dark current). 
    #         This will get rid of dark-current noise.
    
    log.debug('Getting rid of dark-current noise')
    medians = func.remove_dark(medians)

    #cv.imshow("no noise", medians[0])


    # Step 3: Find the global, maximum pixel value of all the images, p_max, 
    #         and normalize all your images to that value (p_max = white).
     
    max_pxl = func.max_pixel(medians)
    log.debug('Max pixel value is: {0}'.format(max_pxl))

    log.debug('Normalizing the images.')
    normalized = func.normalize_images(medians, max_pxl)

    cv.imshow("normalized 0", normalized[0])

    # Step 4: Apply logarithm to the image. 
    #         This will bring dark cells up to level (with poor contrast).
    
    log.debug('Applying logarithm tranformation.')
    log_imgs = func.logarithm(normalized)
    cv.imshow("log 0", log_imgs[0])

    # Step 5: Apply some crazy-bananas contrast correction. 
    #         Like +40 or +50. This fixes the poor contrast caused by the logarithm.
    #         Since we can't apply contrast to 16-bit images, we will
    #         first convert the images to 8-bit.
    
    log.debug('Converting Images to 8-bit')
    converted_imgs = func.convertTo8bit(log_imgs)

    log.debug('Applying Contrast Transform with alpha={0}'.format(1.35) )

    contrast_imgs = func.changeContrastPillow(converted_imgs, 1.35)
    cv.imshow("con 2", contrast_imgs[0])
    

    # Step 6:  Now do some extensive Opening. 
    #         I usually do this by applying 2-3 plain erosions (erode) 
    #         followed by the same number of dilations (dilate). 
    

    kernelSize = 7
    iterations = 5
    kernel = np.ones((kernelSize,kernelSize), np.uint8)

    log.debug('Applying Erode with kernel size: {0} and iterations: {1}'.format(kernelSize, iterations) )
    erode_imgs = func.erode(contrast_imgs, kernel, iterations)
    cv.imshow('erode 1', erode_imgs[0])


    log.debug('Applying Dilation with kernel size: {0} and iterations: {1}'.format(kernelSize, iterations) )
    dilate_imgs = func.dilation(erode_imgs, kernel, iterations)
    cv.imshow('dila 1', dilate_imgs[0])
    

    # Step 7:  Apply local thresholding. 
    #          We use NIBlack threshold.
    

    block = 11
    k = 1

    log.debug('Applying NIBlack with window size: {0} and k: {1}'.format(block, k))
    niblack_imgs = func.niBlackThreshold(dilate_imgs, blockSize=block, k=k)
    cv.imshow('thresh 1', niblack_imgs[0])


    # Step 8:  Finding Contours 
    

    log.debug('Finding Contours')
    img1color=cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img1color2=cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    contours = func.findContours(niblack_imgs)
    cv.drawContours(img1color, contours[0], -1, (0,2**16,0),3)
    cv.imshow('cont all', img1color*5)
    
    accepted_cnts = func.quadriterals(contours)

    cv.drawContours(img1color2, accepted_cnts[0], -1, (0,2**16,0),3)
    cv.imshow('cont filter', img1color2*5)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    log.basicConfig(
        level=log.DEBUG,
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    run()

