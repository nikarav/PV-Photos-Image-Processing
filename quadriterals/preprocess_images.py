import cv2
import os 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# put in your path to github folder
to_github = r"C:\Users\morte\Documents\GitHub"
image_folder = r"C:\Users\morte\Desktop\Test_6_1\Pos0"
save_folder = r"C:\Users\morte\Desktop\Test_6_1\pre-processed"

def normalize_images(image_folder, save_folder):
    '''
    Will normalize the images from image_folder and save them in save_folder,
    which should be already existing!

    '''
    global_max = 0
    # Your images should in any case be loaded as int16 (not uint16).
    for image_name in os.listdir(image_folder):
        if ".tif" not in image_name:
            continue
        path = os.path.join(image_folder, image_name)
        img_gray = cv.imread(path, -1)
        
        global_max = max(global_max, np.max(img_gray))
    
    for image_name in os.listdir(image_folder):
        if ".tif" not in image_name:
            continue
        path = os.path.join(image_folder, image_name)
        img_gray = cv.imread(path, -1)
        # apply some conservative filter
        img_gray[img_gray < 0] = 0   
        # scale image so global max equals total white
        new_image = ((img_gray - 0) * (1/(global_max - 0) * 2**16)).astype('uint16')
        save_path = os.path.join(save_folder, image_name)
        cv2.imwrite(save_path, new_image) 
        
normalize_images(image_folder, save_folder)