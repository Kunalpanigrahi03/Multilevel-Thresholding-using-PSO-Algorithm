# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:23:52 2024

@author: KIIT
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def makeHist(img):
    row, col = img.shape # img is a grayscale image
    y = np.zeros((256), np.uint64)
    for i in range(row):
        for j in range(col):
            y[img[i, j]] += 1
    x = np.arange(0, 256)
    plt.figure(2)
    plt.bar(x, y, color="gray", align="center")
    plt.show()

def main():
    img = cv.imread(r"C:\Users\KIIT\OneDrive\Desktop\Major1\lena.tif", cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found or could not be loaded.")
        return
    
    # Display the image using matplotlib
    plt.figure(1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()
    
    makeHist(img)

if __name__ == "__main__":
    main()
