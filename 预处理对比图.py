import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter


# -------------------------- 1. 预处理函数 --------------------------
def adjust_light_and_color(image):
    """Light and color adjustment"""
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # CLAHE brightness equalization
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(10, 10))
    cl = clahe.apply(l)
    lab_adjusted = cv2.merge((cl, a, b))
    img_cv = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

    # HSV color adjustment
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    img_cv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def preprocess_steps(image_path):
    """Get all preprocessing step results for a single image"""
    # Read original image
    original = Image.open(image_path).convert("RGB")

    # Step 1: Light and color adjustment
    light_color = adjust_light_and_color(original)

    # Step 2: Denoising (Median Filter)
    denoise = light_color.filter(ImageFilter.MedianFilter(size=3))

    # Step 3: Detail enhancement (Sharpening)
    enhancer = ImageEnhance.Sharpness(denoise)
    detail = enhancer.enhance(1.8)

    return {
        "Original Image": original,
        "Light & Color Adjustment": light_color,
        "Denoising": denoise,
        "Detail Enhancement": detail
    }


# -------------------------- 2. 可视化辅助函数 --------------------------
def plot_histogram(image, ax):
    """Plot RGB three-channel histogram"""
    img_np = np.array(image)
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, alpha=0.7)
    ax.set_xlim([0, 256])
    ax.set_ylim([0, max(ax.get_ylim()) * 1.1])  # Leave top space


def plot_edge_detection(image, ax):
    """Plot edge detection results (Canny)"""
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)  # Canny edge detection
    ax.imshow(edges, cmap='gray')
    ax.axis('off')


# -------------------------- 3. 主可视化函数 --------------------------
def visualize_preprocess_single(image_path, save_path=None):
    """Visualize preprocessing steps for a single image"""
    # Get images of each step
    steps = preprocess_steps(image_path)
    step_names = list(steps.keys())
    col_labels = ['a', 'b', 'c', 'd']  # Column labels
    img_name = os.path.basename(image_path)

    # Set plot layout (3 rows x 4 columns: Image → Histogram → Edge Detection)
    # Remove Chinese font settings since we're using English now
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Image Processing Steps', fontsize=16, fontweight='bold')

    # Row 1: Original image + processed images of each step
    for col, name in enumerate(step_names):
        ax = axes[0, col]
        ax.imshow(steps[name])
        ax.set_title(name, fontsize=12)
        ax.axis('off')

    # Row 2: Histograms of each step
    for col, name in enumerate(step_names):
        ax = axes[1, col]
        plot_histogram(steps[name], ax)
        ax.set_title(f'{name} - Histogram', fontsize=12)

    # Row 3: Edge detection of each step
    for col, (name, label) in enumerate(zip(step_names, col_labels)):
        ax = axes[2, col]
        plot_edge_detection(steps[name], ax)
        ax.set_title(f'{name} - Edge Detection', fontsize=12)
        # Add column label below edge detection plot
        ax.text(0.5, -0.1, label, transform=ax.transAxes,
                ha='center', va='top', fontsize=14, fontweight='bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization result saved to: {save_path}")
    plt.show()


# -------------------------- 4. 运行示例 --------------------------
if __name__ == "__main__":
    input_image_path = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//val//00054.jpg"
    save_output_path = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//可视化文件//00054.jpg"
    # Generate visualization
    visualize_preprocess_single(input_image_path, save_output_path)