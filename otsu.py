import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import time
from memory_profiler import memory_usage
import os

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

class OtsuMultiThreshold:
    def __init__(self, image_path):
        supported_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in supported_extensions:
            raise ValueError(f"Unsupported file format. Supported formats: {supported_extensions}")
        if file_ext in ['.tif', '.tiff']:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = np.array(Image.open(image_path).convert("L"))
        self.original_image = image
        self.threshold_values = {}
        self.histogram = self.calculate_histogram(image)

    def calculate_histogram(self, img):
        row, col = img.shape 
        histogram = np.zeros(256)
        for i in range(row):
            for j in range(col):
                histogram[img[i,j]] += 1
        return histogram

    def plot_histogram(self):
        x = np.arange(0, 256)
        plt.figure(figsize=(10, 5))
        plt.bar(x, self.histogram, color='b', width=5, align='center', alpha=0.25)
        plt.title('Image Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def weight(self, start, end):
        return np.sum(self.histogram[start:end])

    def mean(self, start, end):
        w = self.weight(start, end)
        if w == 0:
            return 0
        return np.sum(np.arange(start, end) * self.histogram[start:end]) / w

    def variance(self, start, end):
        m = self.mean(start, end)
        w = self.weight(start, end)
        if w == 0:
            return 0
        return np.sum(((np.arange(start, end) - m) ** 2) * self.histogram[start:end]) / w

    def calculate_multi_thresholds(self, num_thresholds=1):
        start_time = time.time()
        def process_thresholding():
            total_pixels = np.sum(self.histogram)
            thresholds = []
            variance_values = {}
            for i in range(1, 256):
                wb = self.weight(0, i) / total_pixels
                mb = self.mean(0, i)
                vb = self.variance(0, i)
                wf = self.weight(i, 256) / total_pixels
                mf = self.mean(i, 256)
                vf = self.variance(i, 256)
                V2b = wb * wf * (mb - mf)**2
                V2w = wb * vb + wf * vf
                variance_values[i] = V2b / V2w if V2w != 0 else 0
            sorted_variances = sorted(variance_values.items(), key=lambda x: x[1], reverse=True)
            thresholds = [t[0] for t in sorted_variances[:num_thresholds]]
            thresholds.sort()
            return thresholds
        mem_usage = memory_usage(process_thresholding, max_usage=True)
        optimal_thresholds = process_thresholding()
        end_time = time.time()
        execution_time = end_time - start_time
        return optimal_thresholds, execution_time, mem_usage

    def apply_thresholds(self, thresholds):
        segmented_image = np.zeros_like(self.original_image)
        num_regions = len(thresholds) + 1
        thresholds = [0] + sorted(thresholds) + [255]
        for i in range(num_regions):
            if i == 0:
                mask = self.original_image < thresholds[i+1]
            elif i == num_regions - 1:
                mask = self.original_image >= thresholds[i]
            else:
                mask = (self.original_image >= thresholds[i]) & (self.original_image < thresholds[i+1])
            segmented_image[mask] = int(255 / (num_regions-1)) * i
        return segmented_image

def main(image_path):
    otsu = OtsuMultiThreshold(image_path)
    otsu.plot_histogram()
    threshold_levels = [1, 2, 3, 4]
    results = []
    for num_thresholds in threshold_levels:
        optimal_thresholds, execution_time, memory_usage = otsu.calculate_multi_thresholds(num_thresholds)
        segmented_image = otsu.apply_thresholds(optimal_thresholds)
        segmentation_quality = psnr(otsu.original_image, segmented_image)
        results.append({
            'num_thresholds': num_thresholds + 1,
            'optimal_thresholds': optimal_thresholds,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'segmentation_quality': segmentation_quality,
            'segmented_image': segmented_image
        })
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(otsu.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    for i, result in enumerate(results, 1):
        plt.subplot(1, len(results) + 1, i + 1)
        plt.imshow(result['segmented_image'], cmap='gray')
        plt.title(f'{result["num_thresholds"]} Levels')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("\nMulti-level Thresholding Results:")
    print("-" * 50)
    for result in results:
        print(f"\nThreshold Levels: {result['num_thresholds']}")
        print(f"Optimal Thresholds: {result['optimal_thresholds']}")
        print(f"Execution Time: {result['execution_time']:.4f} seconds")
        print(f"Memory Usage : {result['memory_usage']} MiB")
        print(f"Segmentation Quality (PSNR): {result['segmentation_quality']:.2f} dB")

if __name__ == "__main__":
    image_path = "tire.tif"  
    main(image_path)