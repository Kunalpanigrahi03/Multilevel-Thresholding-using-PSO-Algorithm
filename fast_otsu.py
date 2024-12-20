import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage
import itertools

class OtsuMultilevelThresholding:
    def __init__(self, image):
        self.image = image
        self.histogram = self._calculate_histogram()
        self.total_pixels = np.sum(self.histogram)
        self.total_mean = np.sum(np.arange(256) * self.histogram) / self.total_pixels

    def _calculate_histogram(self):
        hist = np.zeros(256, dtype=np.int32)
        for pixel in self.image.ravel():
            hist[pixel] += 1
        return hist

    def _calculate_omegas_and_means(self, thresholds):
        classes = []
        last_threshold = 0
        for threshold in thresholds + [255]:
            class_pixels = self.histogram[last_threshold:threshold+1]
            class_prob = np.sum(class_pixels) / self.total_pixels
            class_mean = np.sum(np.arange(last_threshold, threshold+1) * class_pixels) / np.sum(class_pixels) if np.sum(class_pixels) > 0 else 0
            classes.append({
                'probability': class_prob,
                'mean': class_mean
            })
            last_threshold = threshold + 1
        return classes

    def calculate_between_class_variance(self, thresholds):
        classes = self._calculate_omegas_and_means(thresholds)
        variance = 0
        global_mean = self.total_mean
        for cls in classes:
            variance += cls['probability'] * (cls['mean'] - global_mean)**2
        return variance

    def find_optimal_thresholds(self, num_thresholds):
        best_thresholds = None
        max_variance = 0
        candidates = list(range(1, 255, max(1, 255 // (num_thresholds * 10))))
        for thresholds in itertools.combinations(candidates, num_thresholds):
            variance = self.calculate_between_class_variance(list(thresholds))
            if variance > max_variance:
                max_variance = variance
                best_thresholds = list(thresholds)
        return best_thresholds or [128]

    def apply_thresholds(self, thresholds):
        sorted_thresholds = sorted(thresholds)
        full_thresholds = [0] + sorted_thresholds + [255]
        segmented_image = np.zeros_like(self.image)
        for i in range(1, len(full_thresholds)):
            mask = (self.image >= full_thresholds[i-1]) & (self.image < full_thresholds[i])
            segmented_image[mask] = int(255 / (len(full_thresholds)-1)) * (i-1)
        return segmented_image

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def multilevel_otsu(image_path, num_thresholds):
    start_time = time.time()
    def process_image():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found!")
        otsu = OtsuMultilevelThresholding(image)
        thresholds = otsu.find_optimal_thresholds(num_thresholds)
        segmented_image = otsu.apply_thresholds(thresholds)
        return image, segmented_image, thresholds
    mem_usage = memory_usage(process_image, max_usage=True, timeout=30)
    image, segmented_image, thresholds = process_image()
    end_time = time.time()
    execution_time = end_time - start_time
    segmentation_quality = psnr(image, segmented_image)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    hist, _ = np.histogram(image.ravel(), bins=256, range=[0, 256])
    plt.plot(hist, color='black')
    plt.title(f'Histogram (Thresholds: {thresholds})')
    for t in thresholds:
        plt.axvline(x=t, color='r', linestyle='--')
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image, cmap='gray')
    plt.title(f'Segmented Image ({num_thresholds + 1} levels)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"\n--- Performance Metrics for {num_thresholds + 1}-level Thresholding ---")
    print(f"Execution Time: {execution_time:.4f} seconds")
    print(f"Memory Usage: {mem_usage:.2f} MiB")
    print(f"Segmentation Quality (PSNR): {segmentation_quality:.2f} dB")
    print(f"Thresholds: {thresholds}")
    return execution_time, mem_usage, segmentation_quality

def main():
    image_path = 'tire.tif'
    results = []
    threshold_levels = [1, 2, 3, 4]
    for num_thresholds in threshold_levels:
        print(f"\n{num_thresholds + 1}-level Thresholding:")
        result = multilevel_otsu(image_path, num_thresholds)
        results.append({
            'levels': num_thresholds + 1,
            'execution_time': result[0],
            'memory_usage': result[1],
            'segmentation_quality': result[2]
        })
    print("\n--- Summary of Multilevel Thresholding Performance ---")
    print("Levels | Execution Time (s) | Memory Usage (MiB) | PSNR (dB)")
    print("-" * 60)
    for result in results:
        print(f"{result['levels']:6d} | {result['execution_time']:17.4f} | {result['memory_usage']:17.2f} | {result['segmentation_quality']:14.2f}")

if __name__ == "__main__":
    main()