import os
import cv2
import matplotlib.pyplot as plt 
import imutils
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp


# put in your path to github folder
to_github = r"C:\Users\morte\Documents\GitHub"
image_folder = to_github + "\ImageProcessing\Data\Test_2_1\Pos0"
dark_folder = to_github + "\ImageProcessing\Data\Test_2_dark\Pos0"
save_folder = to_github + "ImageProcessing\Data\Images_2" # if we want to
# save the images where that are dark from the normal ones.


def threshold_gray(img, threshold = 0):
    '''
    Removes the black outline around the cell if it "exists".
    Increase the threshold to remove more than just pure black but also
    slightly gray.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

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

def load_images(image_folder, IMG_CONSTANT = 5, threshold = False):
    images = []
    for image in os.listdir(image_folder):
        if '.tif' not in image:
            continue
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path) * IMG_CONSTANT
        if threshold:
            img = threshold_gray(img, threshold=10)
        images.append(img)
    return images

images = load_images(image_folder, threshold=False)   


stitcher = cv2.Stitcher_create(1)
status, result = stitcher.stitch(images)

plt.figure(dpi=300)
plt.axis('off')
plt.imshow(result) 


plt.figure(dpi=300)
plt.axis('off')
plt.imshow(threshold_gray(result, threshold=20)) 
# https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/