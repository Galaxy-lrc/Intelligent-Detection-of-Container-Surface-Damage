import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
from tqdm import tqdm
import json
import warnings
import cv2
import random
import math
import pandas as pd

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import cm
    from matplotlib.patches import Rectangle, Polygon

    HAS_VISUALIZATION_LIBS = True
except ImportError:
    HAS_VISUALIZATION_LIBS = False
    warnings.warn("未安装 matplotlib 或 seaborn，将跳过可视化。")

# ========== 1：设置中文字体 ==========
if HAS_VISUALIZATION_LIBS:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加中文字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'  # 解决暗色背景问题

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 路径定义
train_img_dir = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//final_train"
val_img_dir = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//final_val"
annotations_dir = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//Annotations"
output_dir = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//问题一分场景训练结果集"
scene_processed_dir = os.path.join(output_dir, "scene_processed_images")
rain_reflection_dir = os.path.join(scene_processed_dir, "rain_reflection")
os.makedirs(rain_reflection_dir, exist_ok=True)
os.makedirs(scene_processed_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ========== 2：定义全局的 collate_fn 函数 ==========
# 全局归一化变换
global_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def train_collate_fn(batch):
    """训练集 collate_fn - 全局定义，确保多进程可访问"""
    images, labels, img_names = zip(*batch)
    images = [global_normalize(img) for img in images]
    return torch.stack(images), torch.stack(labels), img_names


def val_collate_fn(batch):
    """验证集 collate_fn - 全局定义，确保多进程可访问"""
    images, labels, img_names = zip(*batch)
    images = [global_normalize(img) for img in images]
    return torch.stack(images), torch.stack(labels), img_names


# ========================
# 地面反光检测模块
# ========================
class GroundReflectionDetector:
    """地面反光检测器 - 专门识别和处理雨天地面反光"""

    def __init__(self):
        self.reflection_threshold = 0.65  # 反光区域的亮度阈值

    def detect_reflection_regions(self, image_array):
        """
        检测图像中的地面反光区域

        参数:
            image_array: numpy数组格式的图像 (RGB或灰度)

        返回:
            reflection_mask: 二值掩码，反光区域为255
            reflection_regions: 反光区域列表 [(x1, y1, x2, y2), ...]
            ground_mask: 地面区域掩码
        """
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        else:
            gray = image_array
            hsv = None

        height, width = gray.shape

        # 1. 初步亮度阈值检测
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 2. 地面区域假设：通常位于图像下方
        ground_region_height = int(height * 0.4)  # 假设地面占图像的40%
        ground_mask = np.zeros((height, width), dtype=np.uint8)
        ground_mask[height - ground_region_height:, :] = 255

        # 3. 在地面区域内检测高亮区域
        ground_bright_mask = cv2.bitwise_and(bright_mask, ground_mask)

        # 4. 形态学操作增强反光区域
        kernel = np.ones((5, 5), np.uint8)
        ground_bright_mask = cv2.morphologyEx(ground_bright_mask, cv2.MORPH_CLOSE, kernel)
        ground_bright_mask = cv2.morphologyEx(ground_bright_mask, cv2.MORPH_OPEN, kernel)

        # 5. 轮廓检测
        contours, _ = cv2.findContours(ground_bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. 过滤和提取反光区域
        reflection_regions = []
        min_area = width * height * 0.001  # 最小面积阈值

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # 计算区域的亮度特征
                region_brightness = np.mean(gray[y:y + h, x:x + w])
                if region_brightness > 200:  # 高亮区域
                    reflection_regions.append((x, y, x + w, y + h))

        # 7. 创建精细化的反光掩码
        reflection_mask = np.zeros((height, width), dtype=np.uint8)
        for (x1, y1, x2, y2) in reflection_regions:
            reflection_mask[y1:y2, x1:x2] = 255

        # 8. 如果HSV可用，检测饱和度（反光区域通常饱和度低）
        if hsv is not None:
            saturation = hsv[:, :, 1]
            _, low_sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY_INV)
            reflection_mask = cv2.bitwise_and(reflection_mask, low_sat_mask)

        return reflection_mask, reflection_regions, ground_mask

    def analyze_reflection_intensity(self, image_array, reflection_mask):
        """分析反光强度分布"""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        # 提取反光区域的像素值
        reflection_pixels = gray[reflection_mask == 255]
        non_reflection_pixels = gray[reflection_mask == 0]

        if len(reflection_pixels) == 0:
            return {
                'reflection_mean': 0,
                'non_reflection_mean': np.mean(non_reflection_pixels) if len(non_reflection_pixels) > 0 else 0,
                'intensity_ratio': 0,
                'reflection_area_ratio': 0
            }

        reflection_mean = np.mean(reflection_pixels)
        non_reflection_mean = np.mean(non_reflection_pixels) if len(non_reflection_pixels) > 0 else 0

        intensity_ratio = reflection_mean / (non_reflection_mean + 1e-6)
        reflection_area_ratio = len(reflection_pixels) / (gray.shape[0] * gray.shape[1])

        return {
            'reflection_mean': float(reflection_mean),
            'non_reflection_mean': float(non_reflection_mean),
            'intensity_ratio': float(intensity_ratio),
            'reflection_area_ratio': float(reflection_area_ratio)
        }

    def visualize_reflection_detection(self, image_pil, save_path=None):
        """可视化反光检测结果"""
        if not HAS_VISUALIZATION_LIBS:
            return

        image_array = np.array(image_pil)
        reflection_mask, reflection_regions, ground_mask = self.detect_reflection_regions(image_array)
        intensity_stats = self.analyze_reflection_intensity(image_array, reflection_mask)

        # 创建可视化图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原始图像
        axes[0, 0].imshow(image_pil)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        # 2. 灰度图像
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('灰度图像')
        axes[0, 1].axis('off')

        # 3. 地面区域掩码
        axes[0, 2].imshow(ground_mask, cmap='hot')
        axes[0, 2].set_title('地面区域检测')
        axes[0, 2].axis('off')

        # 4. 反光区域掩码
        axes[1, 0].imshow(reflection_mask, cmap='Blues')
        axes[1, 0].set_title('反光区域检测')
        axes[1, 0].axis('off')

        # 5. 反光区域叠加
        overlay = image_array.copy()
        overlay[reflection_mask == 255] = [255, 0, 0]  # 红色标记反光区域
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('反光区域标记(红色)')
        axes[1, 1].axis('off')

        # 6. 统计信息
        stats_text = f"反光分析统计:\n"
        stats_text += f"反光区域亮度均值: {intensity_stats['reflection_mean']:.1f}\n"
        stats_text += f"非反光区域亮度均值: {intensity_stats['non_reflection_mean']:.1f}\n"
        stats_text += f"亮度比: {intensity_stats['intensity_ratio']:.2f}\n"
        stats_text += f"反光面积占比: {intensity_stats['reflection_area_ratio']:.3f}\n"
        stats_text += f"检测到反光区域数: {len(reflection_regions)}"

        axes[1, 2].text(0.5, 0.5, stats_text,
                        ha='center', va='center', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 2].axis('off')

        plt.suptitle('地面反光检测分析', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()

        return None


# ========================
# 场景分类模块（增强版，特别关注雨天）
# ========================
class EnhancedSceneClassifier:
    """增强版场景分类器，特别关注雨天和地面反光"""

    def __init__(self):
        self.scene_types = ["sunny", "cloudy", "night", "rainy"]
        self.reflection_detector = GroundReflectionDetector()

    def classify_with_reflection(self, image_pil):
        """
        分类场景，特别考虑地面反光特征

        返回: (scene_type, has_reflection, reflection_intensity)
        """
        img_array = np.array(image_pil)

        # 1. 检测地面反光
        reflection_mask, reflection_regions, _ = self.reflection_detector.detect_reflection_regions(img_array)
        intensity_stats = self.reflection_detector.analyze_reflection_intensity(img_array, reflection_mask)

        has_reflection = intensity_stats['reflection_area_ratio'] > 0.01  # 反光面积超过1%
        reflection_intensity = intensity_stats['intensity_ratio']

        # 2. 基础场景分类
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        else:
            gray = img_array
            hsv = None

        # 基础特征
        mean_brightness = np.mean(gray)
        contrast = np.std(gray)

        if hsv is not None:
            saturation = np.mean(hsv[:, :, 1])
        else:
            saturation = 0

        # 边缘密度（雨天可能模糊）
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # 3. 决策逻辑（优先考虑反光特征）
        if has_reflection and reflection_intensity > 1.5:
            # 有明显反光，很可能是雨天
            scene_type = "rainy"
        elif mean_brightness < 50 and contrast < 40:
            scene_type = "night"
        elif mean_brightness > 180 and saturation > 100 and contrast > 60:
            scene_type = "sunny"
        elif edge_density < 0.01 and contrast < 50:
            scene_type = "rainy"  # 模糊且对比度低可能是雨天
        elif mean_brightness > 100 and mean_brightness < 180 and saturation < 80:
            scene_type = "cloudy"
        else:
            # 默认根据亮度判断
            if mean_brightness < 80:
                scene_type = "night"
            elif mean_brightness > 160:
                scene_type = "sunny"
            else:
                scene_type = "cloudy"

        return scene_type, has_reflection, reflection_intensity


# ========================
# 雨天场景处理器（重点处理地面反光）
# ========================
class RainSceneProcessor:
    """专门处理雨天场景，特别是地面反光"""

    def __init__(self):
        self.reflection_detector = GroundReflectionDetector()

    def process_rain_with_reflection(self, image_pil):
        """
        针对雨天地面反光的专业处理

        处理策略:
        1. 检测和抑制过强的地面反光
        2. 增强非反光区域的对比度
        3. 恢复反光区域的细节
        4. 全局去模糊和清晰化
        """
        img_array = np.array(image_pil)
        height, width = img_array.shape[:2]

        # 1. 检测反光区域
        reflection_mask, reflection_regions, ground_mask = self.reflection_detector.detect_reflection_regions(img_array)

        # 2. 将图像转换为浮点类型进行处理
        img_float = img_array.astype(np.float32) / 255.0

        # 3. 创建反光区域和非反光区域的掩码
        reflection_mask_bool = reflection_mask > 128
        non_reflection_mask = np.logical_not(reflection_mask_bool)

        # 4. 分离反光区域和非反光区域
        reflection_region = img_float.copy()
        non_reflection_region = img_float.copy()

        # 将非反光区域在反光区域中置为0，反之亦然
        for c in range(3):
            reflection_region[:, :, c][non_reflection_mask] = 0
            non_reflection_region[:, :, c][reflection_mask_bool] = 0

        # 5. 处理非反光区域（增强对比度和细节）
        processed_non_reflection = self._enhance_non_reflection_region(non_reflection_region, non_reflection_mask)

        # 6. 处理反光区域（抑制过亮，恢复细节）
        processed_reflection = self._suppress_reflection_region(reflection_region, reflection_mask_bool)

        # 7. 合并处理后的区域
        result = np.zeros_like(img_float)
        for c in range(3):
            result[:, :, c] = processed_non_reflection[:, :, c] + processed_reflection[:, :, c]

        # 8. 全局去模糊和锐化
        result = self._global_deblur_and_sharpen(result)

        # 9. 确保值在有效范围内并转换回uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(result), reflection_mask, reflection_regions

    def _enhance_non_reflection_region(self, region, mask):
        """增强非反光区域"""
        # 1. 对每个颜色通道单独处理
        enhanced = region.copy()

        for c in range(3):
            channel = region[:, :, c]

            # 2. 只对非反光区域应用CLAHE（对比度受限的自适应直方图均衡化）
            channel_uint8 = (channel * 255).astype(np.uint8)

            # 创建CLAHE对象
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # 对非反光区域应用CLAHE
            channel_masked = np.zeros_like(channel_uint8)
            channel_masked[mask] = channel_uint8[mask]

            # 应用CLAHE
            channel_clahe = clahe.apply(channel_masked)

            # 将结果放回
            enhanced[:, :, c][mask] = channel_clahe[mask].astype(np.float32) / 255.0

        # 3. 轻微提高饱和度
        enhanced_hsv = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        enhanced_hsv = enhanced_hsv.astype(np.float32)

        # 增加饱和度
        saturation_factor = 1.2
        enhanced_hsv[:, :, 1][mask] = np.clip(enhanced_hsv[:, :, 1][mask] * saturation_factor, 0, 255)

        enhanced_hsv = np.clip(enhanced_hsv, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        return enhanced

    def _suppress_reflection_region(self, region, mask):
        """抑制反光区域，恢复细节"""
        suppressed = region.copy()

        # 如果没有反光区域，直接返回
        if not np.any(mask):
            return suppressed

        for c in range(3):
            channel = region[:, :, c]
            channel_masked = channel[mask]

            if len(channel_masked) > 0:
                # 1. 降低亮度（乘以一个系数）
                reduction_factor = 0.7  # 降低到原来的70%
                channel[mask] = channel_masked * reduction_factor

                # 2. 应用轻微的高斯模糊来减少刺眼感
                # 先提取反光区域
                channel_uint8 = (channel * 255).astype(np.uint8)

                # 创建只包含反光区域的图像
                reflection_only = np.zeros_like(channel_uint8)
                reflection_only[mask] = channel_uint8[mask]

                # 应用高斯模糊
                blurred = cv2.GaussianBlur(reflection_only, (5, 5), 1.0)

                # 将模糊后的结果放回
                channel[mask] = blurred[mask].astype(np.float32) / 255.0

        # 3. 轻微增加对比度以恢复一些细节
        for c in range(3):
            channel = suppressed[:, :, c]
            channel_masked = channel[mask]

            if len(channel_masked) > 0:
                # 线性拉伸对比度
                min_val = np.min(channel_masked)
                max_val = np.max(channel_masked)

                if max_val > min_val:
                    channel[mask] = (channel_masked - min_val) / (max_val - min_val)

        return suppressed

    def _global_deblur_and_sharpen(self, img_float):
        """全局去模糊和锐化"""
        img_uint8 = (img_float * 255).astype(np.uint8)

        # 1. 双边滤波（保持边缘的同时去噪）
        deblurred = cv2.bilateralFilter(img_uint8, d=9, sigmaColor=75, sigmaSpace=75)

        # 2. CLAHE增强对比度
        lab = cv2.cvtColor(deblurred, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        deblurred = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # 3. 锐化（使用拉普拉斯滤波器）
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(deblurred, -1, kernel)

        return sharpened.astype(np.float32) / 255.0

    def simulate_rain_reflection(self, image_pil, intensity=0.7):
        """
        模拟雨天地面反光效果（用于数据增强）

        参数:
            image_pil: 原始图像
            intensity: 反光强度 (0.0到1.0)
        """
        img_array = np.array(image_pil)
        height, width = img_array.shape[:2]

        # 创建反光效果
        result = img_array.copy().astype(np.float32)

        # 地面区域（假设为图像下方30%）
        ground_start = int(height * 0.7)
        ground_height = height - ground_start

        # 创建渐变反光效果
        for y in range(ground_start, height):
            # 计算当前位置的强度因子
            pos_factor = (y - ground_start) / ground_height
            reflection_factor = intensity * (1 - pos_factor * 0.5)  # 底部反光更强

            # 添加反光效果
            for c in range(3):
                # 在原始值上添加反光
                reflection_add = reflection_factor * (255 - result[y, :, c])
                result[y, :, c] += reflection_add * 0.3  # 控制反光强度

        # 添加随机光斑
        num_glare_spots = int(width * height * 0.0005)  # 根据图像大小决定光斑数量
        for _ in range(num_glare_spots):
            x = random.randint(0, width - 1)
            y = random.randint(ground_start, height - 1)
            radius = random.randint(3, 15)

            # 创建圆形光斑
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= radius:
                        ny = y + dy
                        nx = x + dx

                        if 0 <= ny < height and 0 <= nx < width:
                            # 计算光斑强度（中心最亮）
                            spot_intensity = 1.0 - (dist / radius)
                            for c in range(3):
                                result[ny, nx, c] = np.clip(
                                    result[ny, nx, c] + 200 * spot_intensity * intensity,
                                    0, 255
                                )

        # 轻微模糊以模拟水面的扩散效果
        ground_region = result[ground_start:, :, :].astype(np.uint8)
        ground_blurred = cv2.GaussianBlur(ground_region, (7, 7), 2.0)
        result[ground_start:, :, :] = ground_blurred

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def visualize_rain_processing(self, original_pil, processed_pil, reflection_mask, reflection_regions, save_path):
        """可视化雨天处理过程"""
        if not HAS_VISUALIZATION_LIBS:
            return

        original_array = np.array(original_pil)
        processed_array = np.array(processed_pil)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原始图像
        axes[0, 0].imshow(original_pil)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        # 2. 处理后图像
        axes[0, 1].imshow(processed_pil)
        axes[0, 1].set_title('处理后图像')
        axes[0, 1].axis('off')

        # 3. 反光区域检测
        axes[0, 2].imshow(reflection_mask, cmap='hot')
        axes[0, 2].set_title('反光区域检测')
        axes[0, 2].axis('off')

        # 4. 反光区域标记
        overlay = original_array.copy()
        for (x1, y1, x2, y2) in reflection_regions:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('反光区域标记')
        axes[1, 0].axis('off')

        # 5. 亮度变化分析
        if len(original_array.shape) == 3:
            orig_gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(processed_array, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original_array
            proc_gray = processed_array

        # 提取反光区域的亮度变化
        if np.any(reflection_mask > 0):
            orig_reflection = orig_gray[reflection_mask > 0]
            proc_reflection = proc_gray[reflection_mask > 0]

            axes[1, 1].hist(orig_reflection, bins=50, alpha=0.5, label='原始', color='blue')
            axes[1, 1].hist(proc_reflection, bins=50, alpha=0.5, label='处理后', color='red')
            axes[1, 1].set_xlabel('亮度值')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].set_title('反光区域亮度变化')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. 处理说明
        processing_text = "雨天反光处理流程:\n\n"
        processing_text += "1. 检测地面反光区域\n"
        processing_text += "2. 分离反光/非反光区域\n"
        processing_text += "3. 增强非反光区域对比度\n"
        processing_text += "4. 抑制过强反光\n"
        processing_text += "5. 恢复反光区域细节\n"
        processing_text += "6. 全局双边滤波+CLAHE+锐化\n\n"
        processing_text += f"检测到反光区域: {len(reflection_regions)}个"

        axes[1, 2].text(0.5, 0.5, processing_text,
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        axes[1, 2].axis('off')

        plt.suptitle('雨天地面反光处理分析', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ========================
# 综合场景处理器
# ========================
class ComprehensiveSceneProcessor:
    """综合场景处理器，包含所有场景处理"""

    def __init__(self):
        self.scene_classifier = EnhancedSceneClassifier()
        self.rain_processor = RainSceneProcessor()
        self.reflection_detector = GroundReflectionDetector()

    def process_image(self, image_pil, scene_type=None):
        """
        处理图像，根据场景类型应用不同的处理策略

        参数:
            image_pil: PIL图像
            scene_type: 指定的场景类型，如果为None则自动检测

        返回:
            processed_image: 处理后的图像
            detected_scene: 检测到的场景类型
            has_reflection: 是否有反光
            reflection_intensity: 反光强度
        """
        # 检测场景和反光
        if scene_type is None:
            scene_type, has_reflection, reflection_intensity = self.scene_classifier.classify_with_reflection(image_pil)
        else:
            # 如果是指定的雨天场景，也检测反光
            if scene_type == "rainy":
                img_array = np.array(image_pil)
                reflection_mask, _, _ = self.reflection_detector.detect_reflection_regions(img_array)
                intensity_stats = self.reflection_detector.analyze_reflection_intensity(img_array, reflection_mask)
                has_reflection = intensity_stats['reflection_area_ratio'] > 0.01
                reflection_intensity = intensity_stats['intensity_ratio']
            else:
                has_reflection = False
                reflection_intensity = 0

        # 根据场景类型处理
        if scene_type == "rainy":
            # 使用专门的雨天处理器
            processed_image, reflection_mask, reflection_regions = self.rain_processor.process_rain_with_reflection(
                image_pil)
        elif scene_type == "sunny":
            # 晴天处理 - CLAHE自适应直方图均衡化
            processed_image = self._process_sunny(image_pil)
        elif scene_type == "night":
            # 夜晚处理 - gamma矫正加CLAHE
            processed_image = self._process_night(image_pil)
        elif scene_type == "cloudy":
            # 阴天处理 - 对比度增强和饱和度增强
            processed_image = self._process_cloudy(image_pil)
        else:
            processed_image = image_pil

        return processed_image, scene_type, has_reflection, reflection_intensity

    def _process_sunny(self, image_pil):
        """晴天处理 - CLAHE自适应直方图均衡化"""
        img_array = np.array(image_pil)

        if len(img_array.shape) == 3:
            # 转换为LAB颜色空间
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # 应用CLAHE到L通道
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # 合并通道并转换回RGB
            enhanced_lab = cv2.merge((cl, a, b))
            img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img_array)

    def _process_night(self, image_pil):
        """夜晚处理 - gamma矫正加CLAHE"""
        img_array = np.array(image_pil)

        if len(img_array.shape) == 3:
            # 1. gamma矫正增强暗部细节
            gamma = 0.8  # gamma值小于1会提亮暗部
            img_array = np.power(img_array / 255.0, gamma) * 255.0
            img_array = img_array.astype(np.uint8)

            # 2. CLAHE增强对比度
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # 应用CLAHE到L通道
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # 合并通道并转换回RGB
            enhanced_lab = cv2.merge((cl, a, b))
            img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

            # 3. 轻微降噪
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

        return Image.fromarray(img_array)

    def _process_cloudy(self, image_pil):
        """阴天处理 - 对比度增强和饱和度增强"""
        img_array = np.array(image_pil)

        if len(img_array.shape) == 3:
            # 1. 增强对比度
            # 转换为LAB颜色空间
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # 对L通道进行直方图均衡化增强对比度
            l_eq = cv2.equalizeHist(l)

            # 2. 增强饱和度
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            # 增加饱和度
            s = cv2.multiply(s, 1.3)
            s = np.clip(s, 0, 255)

            # 合并HSV通道并转换回RGB
            enhanced_hsv = cv2.merge([h, s, v])
            img_array = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

            # 3. 合并对比度增强的结果
            # 将增强对比度后的L通道与原始a,b通道合并
            enhanced_lab = cv2.merge((l_eq, a, b))
            contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

            # 4. 混合原图和增强后的图像（避免过度处理）
            alpha = 0.7
            img_array = cv2.addWeighted(img_array, alpha, contrast_enhanced, 1 - alpha, 0)

        return Image.fromarray(img_array)

    def process_and_visualize_all_scenes(self, image_pil, image_name):
        """为图像应用所有场景处理并保存可视化结果（订正版）"""
        if not HAS_VISUALIZATION_LIBS:
            return

        # 定义场景与对应标题、方法说明
        scene_info = [
            {"type": "sunny", "title": "Original (Sunny Scene)", "method": "Method: CLAHE Adaptive Histogram Equalization"},
            {"type": "cloudy", "title": "Original (Cloudy Scene)", "method": "Method: Contrast + Saturation Enhancement"},
            {"type": "rainy", "title": "Original (Rainy Scene)", "method": "Method: Bilateral Filtering + Sharpening + CLAHE"},
            {"type": "night", "title": "Original (Night Scene)", "method": "Method: Gamma Correction + CLAHE"}
        ]

        # 创建 4行×2列 的布局
        fig, axes = plt.subplots(4, 2, figsize=(12, 20))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        for row_idx, info in enumerate(scene_info):
            scene_type = info["type"]
            # 获取原始图像和处理后图像
            processed, _, _, _ = self.process_image(image_pil.copy(), scene_type)

            # 左列：原始图像
            ax_left = axes[row_idx, 0]
            ax_left.imshow(image_pil)
            ax_left.set_title(info["title"], fontsize=10, pad=5)
            ax_left.axis('off')

            # 右列：处理后图像
            ax_right = axes[row_idx, 1]
            ax_right.imshow(processed)
            ax_right.set_title(f"Processed\n{info['method']}", fontsize=10, pad=5)
            ax_right.axis('off')

            # 保存处理后的图像
            scene_save_path = os.path.join(scene_processed_dir, f'{image_name}_{scene_type}.jpg')
            processed.save(scene_save_path)

        # 全局标题
        plt.suptitle(f'{image_name} - Scene Classification and Adaptive Preprocessing', fontsize=14, y=0.99)
        # 底部总标题
        fig.text(0.5, 0.02, "Figure 6: Scene Classification and Adaptive Preprocessing Flow", ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # 保存对比图
        all_scenes_path = os.path.join(scene_processed_dir, f'{image_name}_all_scenes.png')
        plt.savefig(all_scenes_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 已保存 {image_name} 的场景处理结果到: {all_scenes_path}")

        # 雨天场景额外生成反光分析图表
        rain_info = scene_info[2]
        if rain_info["type"] == "rainy":
            reflection_save_path = os.path.join(rain_reflection_dir, f'{image_name}_reflection_detection.png')
            self.reflection_detector.visualize_reflection_detection(image_pil.copy(), reflection_save_path)

            rain_save_path = os.path.join(rain_reflection_dir, f'{image_name}_rain_processing.png')
            reflection_mask, reflection_regions, _ = self.reflection_detector.detect_reflection_regions(
                np.array(image_pil))
            processed_rain, _, _ = self.rain_processor.process_rain_with_reflection(image_pil.copy())
            self.rain_processor.visualize_rain_processing(
                image_pil, processed_rain, reflection_mask, reflection_regions, rain_save_path
            )

            simulated_reflection = self.rain_processor.simulate_rain_reflection(image_pil.copy(), intensity=0.7)
            simulated_save_path = os.path.join(rain_reflection_dir, f'{image_name}_simulated_reflection.jpg')
            simulated_reflection.save(simulated_save_path)

            print(f"✅ 雨天反光分析已保存到: {rain_reflection_dir}")


# ========================
# 场景增强的Transform
# ========================
class SceneAwareTransform:
    """训练时的场景感知数据增强"""

    def __init__(self, scene_processor, mode='train'):
        self.scene_processor = scene_processor
        self.mode = mode
        # 训练模式：添加 ToTensor，将 PIL 转 Tensor
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()  # 新增：PIL -> Tensor
        ])
        # 验证模式：单独定义（包含 ToTensor）
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()  # 新增：PIL -> Tensor
        ])

    def __call__(self, image):
        if self.mode == 'train':
            # 训练时：随机选择场景并进行处理，然后应用基础增强（含 ToTensor）
            scene_types = ["sunny", "cloudy", "night", "rainy"]
            scene_type = random.choice(scene_types)

            # 如果是雨天，有50%的概率模拟反光增强
            if scene_type == "rainy" and random.random() > 0.5:
                # 使用雨天处理器的模拟反光功能
                reflection_intensity = random.uniform(0.3, 0.9)
                processed = self.scene_processor.rain_processor.simulate_rain_reflection(image, reflection_intensity)
            else:
                processed, _, _, _ = self.scene_processor.process_image(image, scene_type)

            image = self.base_transform(processed)  # 这里已经转为 Tensor
        else:
            # 验证/测试时：根据实际场景处理，再转 Tensor
            processed, _, _, _ = self.scene_processor.process_image(image, None)
            image = self.val_transform(processed)  # 这里已经转为 Tensor

        return image


# ========================
# 解析XML：支持多标签
# ========================
def parse_xml_annotation_multi(xml_file):
    labels = set()
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None and name_elem.text is not None:
                labels.add(name_elem.text.strip())
    except Exception as e:
        print(f"解析XML文件 {xml_file} 时出错: {e}")
    return list(labels) if labels else ["normal"]


def get_all_annotations_multi(annotations_dir):
    annotations = {}
    all_classes = set()
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    print(f"找到 {len(xml_files)} 个XML标注文件")
    for xml_file in xml_files:
        xml_path = os.path.join(annotations_dir, xml_file)
        img_name = os.path.splitext(xml_file)[0] + '.jpg'
        labels = parse_xml_annotation_multi(xml_path)
        annotations[img_name] = labels
        all_classes.update(labels)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    print(f"发现 {len(class_to_idx)} 个类别: {sorted(class_to_idx.keys())}")
    return annotations, class_to_idx


# ========================
# 多标签数据集
# ========================
class MultiLabelContainerDataset(Dataset):
    def __init__(self, img_dir, annotations, class_to_idx, scene_processor, transform=None, mode='train'):
        self.img_dir = img_dir
        self.annotations = annotations
        self.class_to_idx = class_to_idx
        self.scene_processor = scene_processor
        self.transform = transform
        self.mode = mode
        self.img_names = []
        self.label_vectors = []
        supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']
        for img_name in os.listdir(img_dir):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in supported_ext and img_name in annotations:
                self.img_names.append(img_name)
                labels = annotations[img_name]
                vector = torch.zeros(len(class_to_idx), dtype=torch.float32)
                for label in labels:
                    if label in class_to_idx:
                        vector[class_to_idx[label]] = 1.0
                self.label_vectors.append(vector)
        print(f"加载 {len(self.img_names)} 张图像用于多标签训练 (场景处理模式: {mode})")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255))

        label = self.label_vectors[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_name


# ========================
# 创建多标签模型（ResNet50）
# ========================
def create_multilabel_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# ========================
# 评估指标
# ========================
def multilabel_classification_report(y_true, y_pred, class_names):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_classes = len(class_names)
    report = {'classes': {}, 'macro_avg': {}, 'micro_avg': {}}
    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(num_classes):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        support = np.sum(y_true[:, i])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        report['classes'][class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': int(support)
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    report['macro_avg'] = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1-score': np.mean(f1s)
    }
    tp_global = np.sum((y_true == 1) & (y_pred == 1))
    fp_global = np.sum((y_true == 0) & (y_pred == 1))
    fn_global = np.sum((y_true == 1) & (y_pred == 0))
    micro_prec = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0
    micro_rec = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0
    report['micro_avg'] = {
        'precision': micro_prec,
        'recall': micro_rec,
        'f1-score': micro_f1
    }
    return report


def print_multilabel_report(report, class_names):
    print("\n多标签分类报告:")
    print("-" * 80)
    print(f"{'类别':<15} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}")
    print("-" * 80)
    for cls in class_names:
        m = report['classes'][cls]
        print(f"{cls:<15} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<10d}")
    print("-" * 80)
    total_support = sum(report['classes'][cls]['support'] for cls in class_names)
    print(f"macro avg{'':<10} {report['macro_avg']['precision']:<10.4f} {report['macro_avg']['recall']:<10.4f} "
          f"{report['macro_avg']['f1-score']:<10.4f} {total_support:<10d}")
    print(f"micro avg{'':<10} {report['micro_avg']['precision']:<10.4f} {report['micro_avg']['recall']:<10.4f} "
          f"{report['micro_avg']['f1-score']:<10.4f} {total_support:<10d}")
    print("-" * 80)


# ========================
# 新增：绘制并保存混淆矩阵（原始 + 归一化）及 CSV 表格
# ========================
def save_and_plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    if not HAS_VISUALIZATION_LIBS:
        print("Skipping confusion matrix visualization (missing matplotlib/seaborn)")
        return

    # 创建子目录保存 CSV
    cm_csv_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_csv_dir, exist_ok=True)

    n_classes = len(class_names)
    cols = 3
    rows = (n_classes + cols - 1) // cols

    # 准备绘图
    fig_raw, axes_raw = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig_norm, axes_norm = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if n_classes == 1:
        axes_raw = [axes_raw]
        axes_norm = [axes_norm]
    else:
        axes_raw = axes_raw.flatten()
        axes_norm = axes_norm.flatten()

    for i, cls in enumerate(class_names):
        # 计算混淆矩阵
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        cm = np.array([[tn, fp], [fn, tp]])

        # 归一化（按行：真实标签）
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # 避免除零

        # 保存 CSV
        df_raw = pd.DataFrame(cm,
                              index=['Actual Negative', 'Actual Positive'],
                              columns=['Predicted Negative', 'Predicted Positive'])
        df_norm = pd.DataFrame(cm_norm,
                               index=['Actual Negative', 'Actual Positive'],
                               columns=['Predicted Negative', 'Predicted Positive'])
        df_raw.to_csv(os.path.join(cm_csv_dir, f'cm_{cls}_raw.csv'), encoding='utf-8-sig')
        df_norm.to_csv(os.path.join(cm_csv_dir, f'cm_{cls}_normalized.csv'), encoding='utf-8-sig')

        # 绘制原始
        ax_raw = axes_raw[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_raw,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax_raw.set_title(f'Class: {cls}')

        # 绘制归一化
        ax_norm = axes_norm[i]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax_norm,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax_norm.set_title(f'Class: {cls} (Normalized)')

    # 隐藏多余子图
    for j in range(i + 1, len(axes_raw)):
        axes_raw[j].axis('off')
        axes_norm[j].axis('off')

    # 保存图像
    plt.figure(fig_raw.number)
    plt.tight_layout()
    raw_path = os.path.join(output_dir, 'multilabel_confusion_matrices.png')
    plt.savefig(raw_path, dpi=150, bbox_inches='tight')
    plt.close(fig_raw)

    plt.figure(fig_norm.number)
    plt.tight_layout()
    norm_path = os.path.join(output_dir, 'multilabel_confusion_matrices_normalized.png')
    plt.savefig(norm_path, dpi=150, bbox_inches='tight')
    plt.close(fig_norm)

    print(f"✅ 混淆矩阵图像已保存至: {raw_path}")
    print(f"✅ 归一化混淆矩阵图像已保存至: {norm_path}")
    print(f"✅ 混淆矩阵 CSV 表格已保存至: {cm_csv_dir}")


# ========================
# 训练函数
# ========================
def train_multilabel_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                           num_epochs=20, model_save_path='best_multilabel_model.pth', patience=5):
    history = {
        'train_loss': [], 'val_loss': [],
        'val_macro_f1': [], 'val_micro_f1': []
    }
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        model.train()
        running_loss = 0.0
        total_samples = 0
        pbar = tqdm(train_loader, desc='训练', ncols=100)
        for inputs, labels, _ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / total_samples
        history['train_loss'].append(train_loss)

        model.eval()
        running_loss = 0.0
        total_samples = 0
        all_true = []
        all_pred = []
        pbar = tqdm(val_loader, desc='验证', ncols=100)
        with torch.no_grad():
            for inputs, labels, _ in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                all_true.extend(labels.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())
                pbar.set_postfix(loss=loss.item())

        val_loss = running_loss / total_samples
        history['val_loss'].append(val_loss)

        class_names = [str(i) for i in range(len(all_true[0]))]
        report = multilabel_classification_report(all_true, all_pred, class_names)
        macro_f1 = report['macro_avg']['f1-score']
        micro_f1 = report['micro_avg']['f1-score']
        history['val_macro_f1'].append(macro_f1)
        history['val_micro_f1'].append(micro_f1)

        print(f'训练 Loss: {train_loss:.4f} | 验证 Loss: {val_loss:.4f}')
        print(f'Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')

        scheduler.step(macro_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), model_save_path)
            print(f'✓ 保存最佳模型 (Macro F1: {best_f1:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("触发早停")
            break

    return history, best_f1


# ========================
# 可视化场景分布
# ========================
def analyze_and_visualize_scene_distribution(dataset, scene_processor, save_dir):
    """分析数据集中场景分布并可视化"""
    if not HAS_VISUALIZATION_LIBS:
        print("Cannot visualize scene distribution, missing matplotlib/seaborn")
        return

    scene_counts = {"sunny": 0, "cloudy": 0, "night": 0, "rainy": 0}
    scene_brightness = {"sunny": [], "cloudy": [], "night": [], "rainy": []}

    print("Analyzing scene distribution in dataset...")
    for img_name in tqdm(dataset.img_names[:100]):  # 只分析前100张以节省时间
        img_path = os.path.join(dataset.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            scene_type, has_reflection, reflection_intensity = scene_processor.scene_classifier.classify_with_reflection(
                image)
            scene_counts[scene_type] += 1

            # 计算亮度
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            brightness = np.mean(gray)
            scene_brightness[scene_type].append(brightness)
        except:
            continue

    # 可视化场景分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 饼图
    sizes = [scene_counts[scene] for scene in scene_counts]
    labels = [f"{scene}\n({count})" for scene, count in scene_counts.items()]
    colors = ['#FFD700', '#A9A9A9', '#191970', '#4682B4']  # 金色, 深灰, 午夜蓝, 钢蓝

    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Dataset Scene Distribution')

    # 箱线图：各场景的亮度分布
    box_data = [scene_brightness[scene] for scene in scene_counts]
    bp = axes[1].boxplot(box_data, labels=scene_counts.keys(), patch_artist=True)

    # 设置箱线图颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    axes[1].set_title('Brightness Distribution by Scene')
    axes[1].set_ylabel('Average Brightness')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Dataset Scene Analysis', fontsize=14)
    plt.tight_layout()

    distribution_path = os.path.join(save_dir, 'scene_distribution_analysis.png')
    plt.savefig(distribution_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 场景分布分析已保存到: {distribution_path}")

    # 保存场景统计到文件
    stats = {
        "total_images_analyzed": sum(scene_counts.values()),
        "scene_counts": scene_counts,
        "scene_brightness_means": {scene: np.mean(vals) if vals else 0 for scene, vals in scene_brightness.items()}
    }

    stats_path = os.path.join(save_dir, 'scene_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"✅ 场景统计信息已保存到: {stats_path}")

    return scene_counts


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    # ========== 修复3：Windows多进程安全设置 ==========
    if os.name == 'nt':
        # Windows系统上，多进程需要设置spawn模式
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("Windows系统：已设置多进程启动模式为 'spawn'")

    try:
        print("=== 初始化综合场景处理器（包含地面反光处理） ===")
        scene_processor = ComprehensiveSceneProcessor()

        print("=== 加载多标签标注 ===")
        annotations, class_to_idx = get_all_annotations_multi(annotations_dir)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        class_names = [idx_to_class[i] for i in range(num_classes)]
        print(f"类别列表: {class_names}")

        # ========================
        # 处理示例图像（特定图片对比可视化）
        # ========================
        print("\n=== 处理特定示例图像 ===")
        # 在此处指定需要进行场景对比可视化的特定图片
        specific_images = ["00001.jpg", "00006.jpg", "00017.jpg","00028.jpg"
                           , "00004.jpg", "00005.jpg", "00013.jpg", "00027.jpg"
                           , "00096.jpg", "00097.jpg", "00098.jpg", "00099.jpg"
                           , "00101.jpg", "00102.jpg", "00103.jpg"]  # 用户可修改此列表指定特定图片
        
        for img_name in specific_images:
            # 检查图像是否存在于训练集或验证集
            example_paths = [
                os.path.join(train_img_dir, img_name),
                os.path.join(val_img_dir, img_name)
            ]

            found = False
            for path in example_paths:
                if os.path.exists(path):
                    print(f"处理特定图像: {img_name} (来自: {path})")
                    image = Image.open(path).convert('RGB')

                    # 为特定图片生成场景对比可视化
                    scene_processor.process_and_visualize_all_scenes(image, img_name)

                    # 自动场景检测和处理
                    processed, detected_scene, has_reflection, reflection_intensity = scene_processor.process_image(
                        image)

                    print(f"  检测到的场景: {detected_scene}")
                    print(f"  是否有反光: {has_reflection}")
                    print(f"  反光强度比: {reflection_intensity:.2f}")

                    # 保存自动处理结果
                    if detected_scene == "rainy" and has_reflection:
                        save_path = os.path.join(rain_reflection_dir, f'{img_name}_auto_processed.jpg')
                        processed.save(save_path)
                        print(f"  雨天反光处理结果已保存: {save_path}")

                    found = True
                    break

            if not found:
                print(f"警告: 未找到特定图像 {img_name}")

        # ========================
        # 创建场景感知的数据集和变换
        # ========================
        print("\n=== 创建场景感知的数据集 ===")

        # 训练时的变换：包含场景增强
        train_scene_transform = SceneAwareTransform(scene_processor, mode='train')

        # 验证时的变换：根据实际场景处理
        val_scene_transform = SceneAwareTransform(scene_processor, mode='val')

        print("=== 创建数据集 ===")
        train_dataset = MultiLabelContainerDataset(
            train_img_dir, annotations, class_to_idx, scene_processor,
            transform=train_scene_transform, mode='train'
        )

        val_dataset = MultiLabelContainerDataset(
            val_img_dir, annotations, class_to_idx, scene_processor,
            transform=val_scene_transform, mode='val'
        )

        # 分析场景分布
        analyze_and_visualize_scene_distribution(train_dataset, scene_processor, output_dir)

        batch_size = 16
        num_workers = min(4, os.cpu_count() or 4)

        # ========== 使用全局定义的 collate_fn ==========
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            collate_fn=train_collate_fn
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=val_collate_fn
        )

        print("=== 初始化模型 ===")
        model = create_multilabel_model(num_classes)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

        model_save_path = os.path.join(output_dir, 'best_multilabel_model_scene_aware_reflection.pth')

        print("=== 开始场景感知训练（包含地面反光处理） ===")
        history, best_f1 = train_multilabel_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=25, model_save_path=model_save_path, patience=5
        )

        # 保存训练历史
        history_path = os.path.join(output_dir, 'multilabel_training_history_scene_aware_reflection.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"\n=== 评估最佳模型 ===")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        all_true, all_pred = [], []
        all_scenes = []

        with torch.no_grad():
            for inputs, labels, img_names in tqdm(val_loader, desc='评估'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_true.extend(labels.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())

        # 保存评估结果
        eval_results_path = os.path.join(output_dir, 'evaluation_results_scene_aware_reflection.npz')
        np.savez(eval_results_path,
                 y_true=np.array(all_true),
                 y_pred=np.array(all_pred))
        print(f"✅ 评估结果已保存至: {eval_results_path}")

        # 生成分类报告
        report = multilabel_classification_report(all_true, all_pred, class_names)
        print_multilabel_report(report, class_names)

        # 保存报告到文件
        report_txt_path = os.path.join(output_dir, 'multilabel_classification_report_scene_aware_reflection.txt')
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write("场景感知多标签分类报告（包含地面反光处理）\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"最佳模型 Macro F1: {best_f1:.4f}\n\n")
            f.write("各场景处理策略（修改版）:\n")
            f.write("- 晴天: CLAHE自适应直方图均衡化\n")
            f.write("- 阴天: 对比度增强和饱和度增强\n")
            f.write("- 夜晚: gamma矫正加CLAHE\n")
            f.write("- 雨天: 双边滤波加锐化加CLAHE\n\n")

            f.write("地面反光处理特色:\n")
            f.write("1. 检测地面反光区域\n")
            f.write("2. 分离反光/非反光区域\n")
            f.write("3. 增强非反光区域对比度\n")
            f.write("4. 抑制过强反光\n")
            f.write("5. 恢复反光区域细节\n")
            f.write("6. 全局双边滤波+CLAHE+锐化\n\n")

            f.write("详细分类报告:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'类别':<15} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}\n")
            f.write("-" * 80 + "\n")

            for cls in class_names:
                m = report['classes'][cls]
                f.write(
                    f"{cls:<15} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<10d}\n")

            f.write("-" * 80 + "\n")
            total_support = sum(report['classes'][cls]['support'] for cls in class_names)
            f.write(
                f"macro avg{'':<10} {report['macro_avg']['precision']:<10.4f} {report['macro_avg']['recall']:<10.4f} "
                f"{report['macro_avg']['f1-score']:<10.4f} {total_support:<10d}\n")
            f.write(
                f"micro avg{'':<10} {report['micro_avg']['precision']:<10.4f} {report['micro_avg']['recall']:<10.4f} "
                f"{report['micro_avg']['f1-score']:<10.4f} {total_support:<10d}\n")

        # 保存最终模型和映射
        final_model_path = os.path.join(output_dir, 'final_multilabel_model_scene_aware_reflection.pth')
        torch.save(model.state_dict(), final_model_path)

        mapping_path = os.path.join(output_dir, 'multilabel_class_mapping_scene_aware_reflection.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'scene_types': ["sunny", "cloudy", "night", "rainy"],
                'scene_processing_description': {
                    'sunny': 'CLAHE自适应直方图均衡化',
                    'cloudy': '对比度增强和饱和度增强',
                    'night': 'gamma矫正加CLAHE',
                    'rainy': '双边滤波加锐化加CLAHE'
                },
                'rain_reflection_processing': {
                    'detection': '基于亮度阈值和地面位置假设检测反光区域',
                    'enhancement': '对非反光区域应用CLAHE增强对比度',
                    'suppression': '降低反光区域亮度，应用高斯模糊减少刺眼感',
                    'recovery': '对比度拉伸恢复反光区域细节',
                    'global_optimization': '双边滤波+CLAHE+锐化'
                }
            }, f, indent=4, ensure_ascii=False)

        # 保存混淆矩阵图像 + CSV 表格
        save_and_plot_confusion_matrices(np.array(all_true), np.array(all_pred), class_names, output_dir)

        print(f"\n=== 训练完成 ===")
        print(f"✅ 所有输出已成功保存至: {output_dir}")
        print(f"✅ 场景处理示例图像已保存至: {scene_processed_dir}")
        print(f"✅ 雨天反光分析已保存至: {rain_reflection_dir}")
        print(f"✅ 最佳模型: {model_save_path}")
        print(f"✅ 最终模型: {final_model_path}")
        print(f"✅ 分类报告: {report_txt_path}")
        print(f"✅ 场景分布分析: {os.path.join(output_dir, 'scene_distribution_analysis.png')}")

        # 可视化训练历史
        if HAS_VISUALIZATION_LIBS:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 1. 训练/验证损失曲线
            axes[0, 0].plot(history['train_loss'], label='Training Loss', marker='o', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss Value')
            axes[0, 0].set_title('Training and Validation Loss Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. F1分数曲线
            axes[0, 1].plot(history['val_macro_f1'], label='Macro F1', marker='o', linewidth=2, color='orange')
            axes[0, 1].plot(history['val_micro_f1'], label='Micro F1', marker='s', linewidth=2, color='red')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('Validation Set F1 Score Curves')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1.0)

            # 3. 各分类F1分数对比（柱状图）
            class_f1s = [report['classes'][cls]['f1-score'] for cls in class_names]
            axes[1, 0].bar(range(len(class_names)), class_f1s, color='skyblue', edgecolor='navy')
            axes[1, 0].set_xlabel('Classes')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('F1 Score Comparison by Class')
            axes[1, 0].set_xticks(range(len(class_names)))
            axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].set_ylim(0, 1.0)

            # 4. 精度-召回率散点图
            precisions = [report['classes'][cls]['precision'] for cls in class_names]
            recalls = [report['classes'][cls]['recall'] for cls in class_names]
            axes[1, 1].scatter(precisions, recalls, s=100, c=class_f1s, cmap='viridis', alpha=0.7)
            for i, cls in enumerate(class_names):
                axes[1, 1].annotate(cls, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
            axes[1, 1].set_xlabel('Precision')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_title('Precision-Recall Distribution by Class')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(0, 1.0)
            axes[1, 1].set_ylim(0, 1.0)

            # 添加颜色条
            cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
            cbar.set_label('F1 Score')

            plt.suptitle('Scene-Aware Multi-Label Training Results Visualization (with Ground Reflection Processing)', fontsize=16)
            plt.tight_layout()
            train_vis_path = os.path.join(output_dir, 'training_visualization_scene_aware_reflection.png')
            plt.savefig(train_vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Training visualization results saved to: {train_vis_path}")

    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n=== 程序执行结束 ===")



    def visualize_reflection_detection(self, image_pil, save_path=None):
        """可视化反光检测结果"""
        if not HAS_VISUALIZATION_LIBS:
            return
    
        image_array = np.array(image_pil)
        reflection_mask, reflection_regions, ground_mask = self.detect_reflection_regions(image_array)
        intensity_stats = self.analyze_reflection_intensity(image_array, reflection_mask)
    
        # 创建可视化图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
        # 1. 原始图像
        axes[0, 0].imshow(image_pil)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
    
        # 2. 灰度图像
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale Image')
        axes[0, 1].axis('off')
    
        # 3. 地面区域掩码
        axes[0, 2].imshow(ground_mask, cmap='hot')
        axes[0, 2].set_title('Ground Region Detection')
        axes[0, 2].axis('off')
    
        # 4. 反光区域标记
        overlay = original_array.copy()
        for (x1, y1, x2, y2) in reflection_regions:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Reflection Region Marked')
        axes[1, 0].axis('off')
    
        # 5. 亮度变化分析
        if len(original_array.shape) == 3:
            orig_gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(processed_array, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original_array
            proc_gray = processed_array
    
        # 提取反光区域的亮度变化
        if np.any(reflection_mask > 0):
            orig_reflection = orig_gray[reflection_mask > 0]
            proc_reflection = proc_gray[reflection_mask > 0]
    
            axes[1, 1].hist(orig_reflection, bins=50, alpha=0.5, label='Original', color='blue')
            axes[1, 1].hist(proc_reflection, bins=50, alpha=0.5, label='Processed', color='red')
            axes[1, 1].set_xlabel('Brightness Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Brightness Change in Reflection Regions')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
        # 6. 处理说明
        processing_text = "Rainy Reflection Processing Flow:\n\n"
        processing_text += "1. Detect ground reflection regions\n"
        processing_text += "2. Separate reflection/non-reflection regions\n"
        processing_text += "3. Enhance contrast of non-reflection regions\n"
        processing_text += "4. Suppress excessive reflection\n"
        processing_text += "5. Restore reflection region details\n"
        processing_text += "6. Global bilateral filtering + CLAHE + sharpening\n\n"
        processing_text += f"Reflection regions detected: {len(reflection_regions)}"
    
        axes[1, 2].text(0.5, 0.5, processing_text,
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        axes[1, 2].axis('off')
    
        plt.suptitle('Rainy Ground Reflection Processing Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
        # 4. 反光区域掩码
        axes[1, 0].imshow(reflection_mask, cmap='Blues')
        axes[1, 0].set_title('Reflection Region Detection')
        axes[1, 0].axis('off')
    
        # 5. 反光区域叠加
        overlay = image_array.copy()
        overlay[reflection_mask == 255] = [255, 0, 0]  # 红色标记反光区域
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Reflection Region Marked (Red)')
        axes[1, 1].axis('off')
    
        # 6. 统计信息
        stats_text = f"Reflection Analysis Statistics:\n"
        stats_text += f"Reflection Mean Brightness: {intensity_stats['reflection_mean']:.1f}\n"
        stats_text += f"Non-reflection Mean Brightness: {intensity_stats['non_reflection_mean']:.1f}\n"
        stats_text += f"Intensity Ratio: {intensity_stats['intensity_ratio']:.2f}\n"
        stats_text += f"Reflection Area Ratio: {intensity_stats['reflection_area_ratio']:.3f}\n"
        stats_text += f"Number of Reflection Regions: {len(reflection_regions)}"
    
        axes[1, 2].text(0.5, 0.5, stats_text,
                        ha='center', va='center', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 2].axis('off')
    
        plt.suptitle('Ground Reflection Detection Analysis', fontsize=14)
        plt.tight_layout()
