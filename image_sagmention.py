# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:44:39 2024

@author: KIIT
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import psutil
import os

def calculate_psnr(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MiB

def makeHist(img):
    row, col = img.shape
    y = np.zeros((256), np.uint64)
    for i in range(row):
        for j in range(col):
            y[img[i, j]] += 1
    return y

def globalThreshold(img, threshold):
    start_time = time.time()
    start_memory = get_memory_usage()

    segmented_img = np.zeros_like(img)
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i, j] >= threshold:
                segmented_img[i, j] = 255
            else:
                segmented_img[i, j] = 0
    
    end_time = time.time()
    end_memory = get_memory_usage()

    # Calculate performance metrics
    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    segmentation_quality = calculate_psnr(img, segmented_img)

    return segmented_img, execution_time, memory_usage, segmentation_quality

def optimalThreshold(img):
    start_time = time.time()
    start_memory = get_memory_usage()

    segmented_img = np.zeros_like(img)
    hist = makeHist(img)
    total_pixels = img.size
    normalized_hist = hist / total_pixels
    
    cumulative_hist2 = np.zeros(len(hist))
    for i in range(len(hist)):
        cumulative_hist2[i] = cumulative_hist2[i-1] + normalized_hist[i] if i > 0 else normalized_hist[i]
    
    cumulative_hist_mean = np.zeros(len(hist))
    for i in range(1, len(hist)):
        cumulative_hist_mean[i] = np.sum(np.arange(i + 1) * normalized_hist[:i + 1])
    
    global_mean = cumulative_hist_mean[-1]
    
    between_class_variance = np.zeros(256)
    for t in range(256):
        weight1 = cumulative_hist2[t]
        weight2 = 1 - weight1
        if weight1 == 0 or weight2 == 0:
            continue
        mean1 = cumulative_hist_mean[t] / weight1
        mean2 = (global_mean - cumulative_hist_mean[t]) / weight2
        between_class_variance[t] = weight1 * weight2 * (mean1 - mean2) ** 2
        
    optimal_threshold = np.argmax(between_class_variance)
    
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i, j] >= optimal_threshold:
                segmented_img[i, j] = 255
            else:
                segmented_img[i, j] = 0
    
    end_time = time.time()
    end_memory = get_memory_usage()

    # Calculate performance metrics
    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    segmentation_quality = calculate_psnr(img, segmented_img)

    return segmented_img, execution_time, memory_usage, segmentation_quality

def main():
    # Read the image
    img = cv.imread(r"C:\Users\KIIT\OneDrive\Desktop\Major1\zelda.tif", cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found or could not be loaded.")
        return
    
    # Display the original image
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    # Display the histogram
    x = np.arange(0, 256)
    y = makeHist(img)
    plt.figure()
    plt.bar(x, y, color="gray", align="center")
    plt.title('Histogram of the Original Image')
    plt.show()
    
    # Global Thresholding
    threshold = 127
    segmented_img1, global_time, global_memory, global_psnr = globalThreshold(img, threshold)
    
    # Optimal Thresholding
    segmented_img2, optimal_time, optimal_memory, optimal_psnr = optimalThreshold(img)
    
    # Display segmented images
    plt.figure()
    plt.imshow(segmented_img1, cmap='gray')
    plt.title('Global Segmented Image')
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.imshow(segmented_img2, cmap='gray')
    plt.title('Optimal Segmented Image')
    plt.axis('off')
    plt.show()

    # Print performance metrics
    print("\n--- Global Thresholding Performance ---")
    print(f"Execution Time: {global_time:.4f} seconds")
    print(f"Memory Usage: {global_memory:.2f} MiB")
    print(f"Segmentation Quality (PSNR): {global_psnr:.2f} dB")

    print("\n--- Optimal Thresholding Performance ---")
    print(f"Execution Time: {optimal_time:.4f} seconds")
    print(f"Memory Usage: {optimal_memory:.2f} MiB")
    print(f"Segmentation Quality (PSNR): {optimal_psnr:.2f} dB")

if __name__ == "__main__":
    main()