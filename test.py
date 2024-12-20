import PySimpleGUI as sg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import entropy
from memory_profiler import memory_usage
import time
import io
import base64

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

class Particle:
    def __init__(self, n_thresholds, img):
        self.position = np.sort(np.random.randint(1, 256, n_thresholds))
        self.velocity = np.zeros(n_thresholds)
        self.best_position = np.copy(self.position)
        self.best_fitness = -float('inf')
        self.img = img

    def evaluate(self, fitness_func):
        fitness = fitness_func(self.img, self.position)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = np.copy(self.position)
        return fitness

def pso(image, n_thresholds=2, n_particles=20, max_iter=100, fitness_func="otsu"):
    w = 0.6
    c1 = 2.0
    c2 = 2.0

    particles = [Particle(n_thresholds, image) for _ in range(n_particles)]
    global_best_position = np.copy(particles[0].position)
    global_best_fitness = -float('inf')
    
    fitness_function = otsu_fitness if fitness_func == "otsu" else kapur_fitness
    fitness_over_time = []
    
    for t in range(max_iter):
        iteration_best_fitness = -float('inf')
        
        for particle in particles:
            fitness = particle.evaluate(fitness_function)
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle.position)

            r1, r2 = np.random.rand(n_thresholds), np.random.rand(n_thresholds)
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (global_best_position - particle.position))
            particle.position = np.clip(np.round(particle.position + particle.velocity), 1, 255).astype(int)
            particle.position = np.sort(particle.position)
            
            iteration_best_fitness = max(iteration_best_fitness, fitness)
        
        fitness_over_time.append(iteration_best_fitness)
    
    return global_best_position, fitness_over_time

def otsu_fitness(img, thresholds):
    thresholds = np.concatenate(([0], thresholds, [256]))
    total_pixels = img.size
    total_mean = img.mean()
    
    sigma_b = 0
    for i in range(len(thresholds) - 1):
        mask = (img >= thresholds[i]) & (img < thresholds[i + 1])
        pixels_in_class = img[mask]
        weight = len(pixels_in_class) / total_pixels
        mean = pixels_in_class.mean() if len(pixels_in_class) > 0 else 0
        
        sigma_b += weight * (mean - total_mean) ** 2
            
    return sigma_b

def kapur_fitness(img, thresholds):
    thresholds = np.concatenate(([0], thresholds, [256]))
    entropy_sum = 0
    
    for i in range(len(thresholds) - 1):
        mask = (img >= thresholds[i]) & (img < thresholds[i + 1])
        hist, _ = np.histogram(img[mask], bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0] 
        ent = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        entropy_sum += ent
    
    return entropy_sum

def apply_threshold(img, thresholds):
    thresholds = np.concatenate(([0], thresholds, [256]))
    segmented_img = np.zeros_like(img)
    num_regions = len(thresholds) - 1
    
    for i in range(num_regions):
        segmented_img[(img >= thresholds[i]) & (img < thresholds[i + 1])] = 255 // num_regions * i
    
    return segmented_img

def create_results_layout(results):
    # Create a table-like layout for results
    headers = ['Method', 'Threshold Levels', 'Execution Time (s)', 'Memory Usage (MiB)', 'Segmentation Quality (dB)']
    
    # Prepare data for the table
    table_data = []
    for result in results:
        table_data.append([
            result['method'].capitalize(),
            result['threshold_levels'],
            f"{result['execution_time']:.4f}",
            f"{result['memory_usage']:.2f}",
            f"{result['segmentation_quality']:.2f}"
        ])
    
    # PySimpleGUI layout
    layout = [
        [sg.Text("Segmentation Method Comparison", font=('Helvetica', 16, 'bold'))],
        [sg.Table(
            values=table_data, 
            headings=headers,
            auto_size_columns=False,
            col_widths=[10, 15, 20, 20, 25],
            justification='center',
            num_rows=min(10, len(table_data)),
            alternating_row_color='lightblue',
            key='-RESULTS_TABLE-'
        )],
        [sg.Button("Close", key='-CLOSE-')]
    ]
    
    return layout

def plot_to_base64(fig):
    # Convert matplotlib figure to base64 for PySimpleGUI
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue())

def display_segmented_images_with_gui(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sg.popup_error("Image not found or path is incorrect")
        return

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    methods = ["otsu", "kapur"]
    threshold_levels = [2, 3, 4, 5]
    segmented_images = []
    fitness_histories = {"otsu": {}, "kapur": {}}
    thresholds_results = {"otsu": {}, "kapur": {}}
    results = []

    # Segmentation and result collection
    for method in methods:
        method_images = []
        for n_thresholds in threshold_levels:
            start_time = time.time()
            mem_usage = memory_usage((lambda: pso(img, n_thresholds=n_thresholds, fitness_func=method)), max_usage=True)
            optimal_thresholds, fitness_over_time = pso(img, n_thresholds=n_thresholds, fitness_func=method)
            execution_time = time.time() - start_time

            segmented_img = apply_threshold(img, optimal_thresholds)
            segmentation_quality = psnr(img, segmented_img)

            method_images.append(segmented_img)
            fitness_histories[method][n_thresholds] = fitness_over_time
            thresholds_results[method][n_thresholds] = optimal_thresholds
            results.append({
                "method": method,
                "threshold_levels": n_thresholds,
                "execution_time": execution_time,
                "memory_usage": mem_usage,
                "segmentation_quality": segmentation_quality,
            })
        segmented_images.append(method_images)

    # Create plots
    fig1, axs = plt.subplots(len(threshold_levels), len(methods), figsize=(12, 16))
    for i, level in enumerate(threshold_levels):
        axs[i, 0].imshow(segmented_images[0][i], cmap="gray")
        axs[i, 0].set_title(f"Otsu - {level} Levels")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(segmented_images[1][i], cmap="gray")
        axs[i, 1].set_title(f"Kapur - {level} Levels")
        axs[i, 1].axis("off")

    plt.tight_layout()

    fig2, ax = plt.subplots(1, 2, figsize=(14, 6))
    for level in threshold_levels:
        ax[0].plot(fitness_histories["otsu"][level], label=f"Threshold {level}")
        ax[1].plot(fitness_histories["kapur"][level], label=f"Threshold {level}")

    ax[0].set_title("Fitness Value vs No of Iterations (Otsu)")
    ax[0].set_xlabel("No of Iterations")
    ax[0].set_ylabel("Fitness Value")
    ax[0].legend()

    ax[1].set_title("Fitness Value vs No of Iterations (Kapur)")
    ax[1].set_xlabel("No of Iterations")
    ax[1].set_ylabel("Fitness Value")
    ax[1].legend()

    plt.tight_layout()

    # Convert plots to base64
    segmentation_plot_base64 = plot_to_base64(fig1)
    fitness_plot_base64 = plot_to_base64(fig2)

    # Create GUI layout
    layout = [
        [sg.Text("Image Segmentation Results", font=('Helvetica', 16, 'bold'))],
        [sg.Image(data=segmentation_plot_base64, key='-SEGMENTATION_PLOT-')],
        [sg.Image(data=fitness_plot_base64, key='-FITNESS_PLOT-')],
        [sg.Column(create_results_layout(results), scrollable=True, vertical_scroll_only=True)],
        [sg.Button("Close", key='-CLOSE-')]
    ]

    # Create the window
    window = sg.Window("Segmentation Analysis", layout, finalize=True, size=(1200, 800))

    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == '-CLOSE-':
            break

    window.close()

if __name__ == "__main__":
    image_path = "lena.tif"
    display_segmented_images_with_gui(image_path)