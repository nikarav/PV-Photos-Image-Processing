import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename != 'metadata.txt':
            img = cv.imread(os.path.join(folder,filename), -1)
            #img = cv.cvtColor(img, cv.COLOR_BAYER_GR2GRAY).astype(np.int16)
            if img is not None:
                #images.append(img.astype(np.int16))
                images.append(img)
    return images


def plot_image(image, title, cmap = 'gray'):
    plt.figure(dpi=300)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def find_max(image):
    return np.amax(image)