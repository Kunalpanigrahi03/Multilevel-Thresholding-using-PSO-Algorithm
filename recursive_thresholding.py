import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
from memory_profiler import memory_usage

def calculate_mean_and_std(image, a, b):
    region = image[(image >= a) & (image <= b)]
    if len(region) == 0:
        return 0, 0
    mean = np.mean(region)
    std = np.std(region)
    return mean, std

def calculate_sub_ranges(mean, std, k1, k2):
    T1 = max(0, mean - k1 * std)
    T2 = min(255, mean + k2 * std)
    return T1, T2

def threshold_image(image, a, T1, T2, b):
    thresholded_image = np.zeros_like(image)
    if T1 > a:
        thresholded_image[(image >= a) & (image <= T1)] = np.mean(image[(image >= a) & (image <= T1)])
    if T2 > T1:
        thresholded_image[(image > T1) & (image <= T2)] = np.mean(image[(image > T1) & (image <= T2)])
    if b > T2:
        thresholded_image[(image > T2) & (image <= b)] = np.mean(image[(image > T2) & (image <= b)])
    return thresholded_image

def recursive_thresholding(image, n, k1, k2):
    a = 0
    b = 255
    for _ in range(int(n / 2)):
        mean, std = calculate_mean_and_std(image, a, b)
        T1, T2 = calculate_sub_ranges(mean, std, k1, k2)
        thresholded_image = threshold_image(image, a, T1, T2, b)
        a = int(T1) + 1
        b = int(T2) - 1
    return thresholded_image

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def analyze_segmentation(original_image, segmented_image):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.subplot(122)
    plt.title('Segmented Image')
    plt.imshow(segmented_image, cmap='gray')
    plt.tight_layout()
    plt.show()
    
    unique_regions = np.unique(segmented_image)
    print(f"\nNumber of Unique Intensity Regions: {len(unique_regions)}")
    print(f"Unique Intensity Regions: {unique_regions}")

def plot_original_histogram(image):
    plt.figure(figsize=(7, 5))
    plt.title('Histogram of Original Image')
    plt.hist(image.flatten(), bins=256, color='gray', edgecolor='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def main():
    image = np.array(Image.open('lena.tif').convert('L'))
    plot_original_histogram(image)
    
    k1 = 1.0
    k2 = 1.0
    threshold_levels = [2, 3, 4, 5]
    results = []
    
    for n in threshold_levels:
        start_time = time.time()
        
        mem_usage = memory_usage((lambda: recursive_thresholding(image.copy(), n, k1, k2)), max_usage=True)
        thresholded_image = recursive_thresholding(image.copy(), n, k1, k2)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        segmentation_quality = psnr(image, thresholded_image)
        
        results.append({
            'threshold_levels': n,
            'execution_time': execution_time,
            'memory_usage': mem_usage,
            'segmentation_quality': segmentation_quality
        })
        
        print(f"\n--- Threshold Levels: {n} ---")
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Memory Usage: {mem_usage:.2f} MiB")
        print(f"Segmentation Quality (PSNR): {segmentation_quality:.2f} dB")
        
        analyze_segmentation(image, thresholded_image)
    
    print("\n--- Summary of Execution Times, Memory Usage, and Segmentation Quality ---")
    for result in results:
        print(f"Threshold Levels: {result['threshold_levels']}, Execution Time: {result['execution_time']:.4f} seconds, Memory Usage: {result['memory_usage']:.2f} MiB, Segmentation Quality (PSNR): {result['segmentation_quality']:.2f} dB")

if __name__ == '__main__':
    main()