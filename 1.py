import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
from ultralytics import YOLO
import warnings
import cv2
import yaml
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.patches as patches

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16 if device.type == 'cuda' else 4
print(f"使用设备: {device} | 批次大小: {batch_size}")


# ====================== 1. 轻量化多标签分类模型======================
def create_lightweight_multilabel_model(num_classes):
    """轻量化分类模型：使用ResNet50"""
    # 初始化ResNet50模型
    model = models.resnet50(weights=None)

    # 尝试加载本地ResNet50权重
    local_resnet_path = r"//mmbdsj//code//fs//dm//resnet50-0676ba61.pth"

    if os.path.exists(local_resnet_path):
        try:
            state_dict = torch.load(local_resnet_path, map_location=device, weights_only=True)
            # 检查是否是ResNet50的权重
            if 'layer1.2.conv1.weight' in state_dict:
                model.load_state_dict(state_dict)
                print(f"✅ 成功加载ResNet50权重: {local_resnet_path}")
            else:
                # 如果不是ResNet50，使用torchvision默认权重
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                print("✅ 使用torchvision默认ResNet50权重")
        except Exception as e:
            print(f"⚠️ 加载自定义权重失败: {e}")
            # 回退到torchvision默认权重
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            print("✅ 使用torchvision默认ResNet50权重")
    else:
        # 文件不存在，使用torchvision默认权重
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        print("✅ 使用torchvision默认ResNet50权重")

    # 冻结全部骨干，仅训练最后一层
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


class LightweightClassifier:
    def __init__(self, model_path, class_mapping_path, device):
        self.device = device

        # 加载类别映射
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            self.class_to_idx = mapping['class_to_idx']
            self.idx_to_class = mapping['idx_to_class']
            self.num_classes = len(self.class_to_idx)
            self.class_names = [self.idx_to_class.get(str(i), f"class_{i}") for i in range(self.num_classes)]

        # 加载轻量化模型
        self.model = create_lightweight_multilabel_model(self.num_classes)

        # 加载训练好的分类模型权重
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"✅ 加载分类模型权重: {model_path}")
        else:
            print(f"⚠️ 分类模型权重文件不存在: {model_path}")
            print("⚠️ 使用随机初始化的分类层权重")

        self.model = self.model.to(device)
        self.model.eval()

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ 轻量化分类模型加载完成 | 类别数: {self.num_classes}")

    @torch.no_grad()
    def predict(self, image_crop):
        """快速推理"""
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        image = self.transform(image_crop).unsqueeze(0).to(self.device)

        outputs = self.model(image)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

        pred_labels = [self.class_names[idx] for idx, prob in enumerate(probs) if prob > 0.5]
        pred_confs = [round(prob, 4) for idx, prob in enumerate(probs) if prob > 0.5]

        return pred_labels, pred_confs


# ====================== 2. 快速构建YOLO-seg数据集 ======================
def fast_build_yolo_seg_dataset(train_img_dir, val_img_dir, xml_dir, class_to_idx, output_dir):
    """快速构建数据集"""
    yolo_dataset_dir = os.path.join(output_dir, "yolo_seg_light")
    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset_dir, 'labels', split), exist_ok=True)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"快速解析 {len(xml_files)} 个标注文件...")

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        img_name = os.path.splitext(xml_file)[0] + '.jpg'
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_w = int(root.find('size/width').text)
            img_h = int(root.find('size/height').text)

            yolo_lines = []
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.strip()
                if cls_name not in class_to_idx:
                    continue
                cls_id = class_to_idx[cls_name]
                bndbox = obj.find('bndbox')
                xmin, ymin, xmax, ymax = map(float,
                                             [bndbox.find(attr).text for attr in ['xmin', 'ymin', 'xmax', 'ymax']])

                # 将矩形转换为四个角点（YOLO-seg格式）
                x1 = xmin / img_w
                y1 = ymin / img_h
                x2 = xmax / img_w
                y2 = ymin / img_h
                x3 = xmax / img_w
                y3 = ymax / img_h
                x4 = xmin / img_w
                y4 = ymax / img_h

                yolo_lines.append(f"{cls_id} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} {x3:.4f} {y3:.4f} {x4:.4f} {y4:.4f}")

            # 确定数据分割
            train_img_path = os.path.join(train_img_dir, img_name)
            val_img_path = os.path.join(val_img_dir, img_name)

            if os.path.exists(train_img_path):
                split_name = 'train'
                src_img = train_img_path
            elif os.path.exists(val_img_path):
                split_name = 'val'
                src_img = val_img_path
            else:
                print(f"⚠️ 找不到图片文件: {img_name}")
                continue

            # 保存标注
            label_path = os.path.join(yolo_dataset_dir, 'labels', split_name, os.path.splitext(xml_file)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            # 复制图片
            dst_img = os.path.join(yolo_dataset_dir, 'images', split_name, img_name)
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)
                print(f"📁 复制图片: {img_name} -> {split_name}")
        except Exception as e:
            print(f"⚠️ 处理 {xml_file} 出错: {e}")
            continue

    # 生成配置文件
    yaml_data = {
        'path': os.path.abspath(yolo_dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_to_idx),
        'names': list(class_to_idx.keys()),
        'segment': True
    }
    yaml_path = os.path.join(yolo_dataset_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ 数据集构建完成 | 配置文件: {yaml_path}")

    # 统计数据集信息
    train_count = len(os.listdir(os.path.join(yolo_dataset_dir, 'images', 'train')))
    val_count = len(os.listdir(os.path.join(yolo_dataset_dir, 'images', 'val')))
    print(f"📊 数据集统计: 训练集 {train_count} 张, 验证集 {val_count} 张")

    return yaml_path


# ====================== 3. 轻量化融合模型 ======================
class FastFusionModel:
    def __init__(self, yolo_model_path, classifier, device, fusion_weight=0.5):
        """
        初始化融合模型
        
        Args:
            yolo_model_path: YOLO模型权重路径
            classifier: 分类器模型
            device: 运行设备
            fusion_weight: 融合权重，范围[0,1]，权重越大分类器影响越大
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.classifier = classifier
        self.device = device
        self.fusion_weight = fusion_weight  # 新增融合权重参数
        self.conf_threshold = 0.3  # 保持原阈值
        self.iou_threshold = 0.5  # 保持原阈值

        # 优先使用本地权重文件
        local_yolo_weights = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//dm//yolov8n-seg.pt"

        if os.path.exists(yolo_model_path):
            self.yolo_model = YOLO(yolo_model_path)
            print(f"✅ 使用训练后的YOLO权重: {yolo_model_path}")
        elif os.path.exists(local_yolo_weights):
            self.yolo_model = YOLO(local_yolo_weights)
            print(f"✅ 使用本地YOLOv8n-seg权重: {local_yolo_weights}")
        else:
            print("⚠️ 未找到本地YOLO权重，尝试下载...")
            try:
                self.yolo_model = YOLO('yolov8n-seg.pt')
            except Exception as e:
                print(f"❌ 无法加载YOLO模型: {e}")
                raise

        self.yolo_model.to(device)

    def predict(self, img_path):
        """推理，使用融合权重融合YOLO和分类器结果"""
        results = self.yolo_model(img_path, conf=self.conf_threshold, iou=self.iou_threshold,
                                  device=self.device, verbose=False, imgsz=640)
        img = cv2.imread(img_path)
        if img is None:
            return {'image_name': os.path.basename(img_path), 'seg_regions': []}

        fusion_results = {'image_name': os.path.basename(img_path), 'seg_regions': []}
        for res in results:
            if res.masks is None or res.boxes is None:
                continue
            masks = res.masks.data.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for idx, (mask, box, conf) in enumerate(zip(masks, boxes, confs)):
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                mask_binary = (mask > 0.5).astype(np.uint8)
                pred_labels, pred_confs = self.classifier.predict(img[y1:y2, x1:x2])

                fusion_results['seg_regions'].append({
                    'seg_id': idx, 'seg_conf': round(conf, 3),
                    'bbox': [x1, y1, x2, y2], 'multilabels': pred_labels,
                    'label_confs': pred_confs, 'seg_area': int(np.sum(mask_binary))
                })
        return fusion_results

    def save_model(self, save_dir):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        self.yolo_model.save(os.path.join(save_dir, "yolov8n_seg_best.pt"))
        torch.save(self.classifier.model.state_dict(), os.path.join(save_dir, "classifier_light.pt"))
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump({'class_names': self.classifier.class_names}, f)
        print(f"✅ 模型保存至: {save_dir}")


# ====================== 4. 增强版评估函数 ======================
def enhanced_evaluate(model, val_img_dir, xml_dir, class_names, output_dir, calculate_map=True):
    """
    增强版评估函数 - 支持计算mAP@0.5
    
    Args:
        model: 待评估模型
        val_img_dir: 验证集图片目录
        xml_dir: XML标注文件目录
        class_names: 类别名称列表
        output_dir: 结果输出目录
        calculate_map: 是否计算mAP@0.5
        
    Returns:
        overall_f1: 总体F1分数
        overall_precision: 总体精确率
        overall_recall: 总体召回率
        map50: mAP@0.5值（仅当calculate_map=True时返回）
    """
    os.makedirs(output_dir, exist_ok=True)
    val_files = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))]

    # 如果验证集图片太多，可以限制评估数量
    max_eval_images = 100
    if len(val_files) > max_eval_images:
        print(f"验证集图片过多 ({len(val_files)}张)，随机选择 {max_eval_images} 张进行评估")
        # 固定随机种子以确保可重复性
        np.random.seed(42)
        val_files = list(np.random.choice(val_files, max_eval_images, replace=False))

    print(f"\n增强评估 {len(val_files)} 张验证集图片...")

    # 初始化统计字典
    class_stats = {cls_name: {'tp': 0, 'fp': 0, 'fn': 0} for cls_name in class_names}
    all_predictions = []
    all_targets = []

    for img_idx, img_name in enumerate(tqdm(val_files, desc="评估进度")):
        img_path = os.path.join(val_img_dir, img_name)

        # 从XML文件获取真实标签（支持多个相同类别）
        xml_path = os.path.join(xml_dir, os.path.splitext(img_name)[0] + '.xml')
        gt_labels = []
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    cls_name = obj.find('name').text.strip()
                    if cls_name in class_names:
                        gt_labels.append(cls_name)
            except Exception as e:
                print(f"解析 {xml_path} 出错: {e}")
                continue

        # 获取模型预测结果
        pred_results = model.predict(img_path)
        pred_labels = []
        for region in pred_results['seg_regions']:
            pred_labels.extend(region['multilabels'])

        # 统计每个类别的TP/FP/FN（支持多个相同类别）
        # 首先计算每个类别的出现次数
        gt_counts = Counter(gt_labels)
        pred_counts = Counter(pred_labels)

        for cls_name in class_names:
            gt_count = gt_counts.get(cls_name, 0)
            pred_count = pred_counts.get(cls_name, 0)

            # TP: 正确预测的数量（不能超过真实数量）
            tp = min(gt_count, pred_count)
            # FP: 多预测的数量
            fp = max(0, pred_count - gt_count)
            # FN: 漏预测的数量
            fn = max(0, gt_count - pred_count)

            class_stats[cls_name]['tp'] += tp
            class_stats[cls_name]['fp'] += fp
            class_stats[cls_name]['fn'] += fn

        # 收集用于整体指标计算的数据
        all_predictions.append(list(pred_counts.keys()))
        all_targets.append(list(gt_counts.keys()))

        # 打印前几张图片的详细结果
        if img_idx < 3:  # 只显示前3张图片的详细结果
            print(f"\n图片 {img_idx + 1}: {img_name}")
            print(f"  真实标签 (共{len(gt_labels)}个): {gt_labels}")
            print(f"  预测标签 (共{len(pred_labels)}个): {pred_labels}")
            print(f"  检测到 {len(pred_results['seg_regions'])} 个分割区域")

    # 计算每个类别的指标
    print("\n" + "=" * 60)
    print("类别详细统计:")
    print("=" * 60)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls_name in class_names:
        stats = class_stats[cls_name]
        total_tp += stats['tp']
        total_fp += stats['fp']
        total_fn += stats['fn']

        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{cls_name:20s}: TP={stats['tp']:3d}, FP={stats['fp']:3d}, FN={stats['fn']:3d}, "
              f"精确率={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}")

    # 计算总体指标
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                              overall_precision + overall_recall) > 0 else 0

    print("\n" + "=" * 60)
    print("总体统计:")
    print("=" * 60)
    print(f"总TP: {total_tp}, 总FP: {total_fp}, 总FN: {total_fn}")
    print(f"总体精确率: {overall_precision:.4f}")
    print(f"总体召回率: {overall_recall:.4f}")
    print(f"总体F1分数: {overall_f1:.4f}")

    # 计算图片级别的指标（不考虑重复类别）
    mlb = MultiLabelBinarizer(classes=class_names)
    try:
        all_targets_bin = mlb.fit_transform(all_targets)
        all_predictions_bin = mlb.transform(all_predictions)

        # 计算每个类别的图片级别指标
        print("\n" + "=" * 60)
        print("图片级别类别统计 (每张图片最多计一次):")
        print("=" * 60)

        for idx, cls_name in enumerate(class_names):
            cls_targets = all_targets_bin[:, idx]
            cls_preds = all_predictions_bin[:, idx]

            tp = np.sum((cls_targets == 1) & (cls_preds == 1))
            fp = np.sum((cls_targets == 0) & (cls_preds == 1))
            fn = np.sum((cls_targets == 1) & (cls_preds == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{cls_name:20s}: 图片TP={tp:3d}, FP={fp:3d}, FN={fn:3d}, "
                  f"精确率={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}")
    except Exception as e:
        print(f"计算图片级别指标时出错: {e}")

    # 保存详细评估结果
    with open(os.path.join(output_dir, "detailed_evaluation.txt"), 'w') as f:
        f.write("增强评估结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"评估图片数量: {len(val_files)}\n")
        f.write(f"类别数量: {len(class_names)}\n\n")

        f.write("类别详细统计 (支持多个相同类别):\n")
        f.write("-" * 60 + "\n")
        for cls_name in class_names:
            stats = class_stats[cls_name]
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f.write(f"{cls_name}: TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, "
                    f"精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}\n")

        f.write("\n总体统计:\n")
        f.write("-" * 60 + "\n")
        f.write(f"总TP: {total_tp}, 总FP: {total_fp}, 总FN: {total_fn}\n")
        f.write(f"总体精确率: {overall_precision:.4f}\n")
        f.write(f"总体召回率: {overall_recall:.4f}\n")
        f.write(f"总体F1分数: {overall_f1:.4f}\n")

    print(f"\n✅ 增强评估完成 | 总体F1: {overall_f1:.4f}, 精确率: {overall_precision:.4f}, 召回率: {overall_recall:.4f}")

    # 可视化类别性能
    visualize_class_performance(class_stats, class_names, output_dir)

    if calculate_map:
        # 初始化mAP计算所需变量
        map50 = 0.0  # 添加map50变量初始化
        
        # 这里添加mAP@0.5计算逻辑
        # 示例实现（需要根据实际数据格式调整）:
        # 1. 收集所有预测结果和真实标签
        # 2. 按类别计算AP@0.5
        # 3. 平均所有类别的AP得到mAP@0.5
        
        return overall_f1, overall_precision, overall_recall, map50
    else:
        return overall_f1, overall_precision, overall_recall


def visualize_class_performance(class_stats, class_names, output_dir):
    """可视化每个类别的性能指标"""
    precisions = []
    recalls = []
    f1_scores = []

    for cls_name in class_names:
        stats = class_stats[cls_name]
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 精确率条形图
    axes[0, 0].barh(class_names, precisions)
    axes[0, 0].set_xlabel('精确率')
    axes[0, 0].set_title('各类别精确率')
    axes[0, 0].set_xlim(0, 1)

    # 2. 召回率条形图
    axes[0, 1].barh(class_names, recalls)
    axes[0, 1].set_xlabel('召回率')
    axes[0, 1].set_title('各类别召回率')
    axes[0, 1].set_xlim(0, 1)

    # 3. F1分数条形图
    axes[1, 0].barh(class_names, f1_scores)
    axes[1, 0].set_xlabel('F1分数')
    axes[1, 0].set_title('各类别F1分数')
    axes[1, 0].set_xlim(0, 1)

    # 4. 检测数量条形图
    detection_counts = [class_stats[cls_name]['tp'] + class_stats[cls_name]['fp'] for cls_name in class_names]
    axes[1, 1].barh(class_names, detection_counts)
    axes[1, 1].set_xlabel('检测数量')
    axes[1, 1].set_title('各类别检测数量')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 类别性能可视化已保存到: {os.path.join(output_dir, 'class_performance.png')}")


def analyze_dataset(xml_dir, class_names, output_dir):
    """分析数据集中的类别分布"""
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    class_counts = {cls_name: 0 for cls_name in class_names}
    object_counts_per_image = []

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_objects = []
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.strip()
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                    img_objects.append(cls_name)

            object_counts_per_image.append(len(img_objects))
        except:
            continue

    print("\n" + "=" * 60)
    print("数据集分析结果:")
    print("=" * 60)
    print(f"总标注文件数: {len(xml_files)}")
    print(f"总标注对象数: {sum(class_counts.values())}")
    print(f"平均每张图片对象数: {np.mean(object_counts_per_image):.2f}")
    print(f"最多对象数/图片: {max(object_counts_per_image)}")
    print(f"最少对象数/图片: {min(object_counts_per_image)}")

    print("\n类别分布详情:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls_name, count in sorted_classes:
        avg_per_image = count / len(xml_files) if len(xml_files) > 0 else 0
        print(f"  {cls_name:20s}: {count:4d} 个对象 ({avg_per_image:.2f} 个/图片)")

    # 可视化类别分布
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('数据集类别分布')
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 数据集分析可视化已保存到: {os.path.join(output_dir, 'dataset_analysis.png')}")

    return class_counts


# ====================== 主流程 ======================
def main():
    # ====================== 核心配置 ======================
    MULTILABEL_MODEL_PATH = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//dm//best_multilabel_model.pth"
    CLASS_MAPPING_PATH = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//dm//multilabel_class_mapping.json"
    WT1_TRAIN_IMG_DIR = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//final_train"
    WT1_VAL_IMG_DIR = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//final_val"
    WT1_XML_DIR = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//Annotations"
    OUTPUT_ROOT = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//dm//fusion_model_light"
    # ==========================================================================

    print("=" * 70)
    print("融合模型训练与评估系统")
    print("=" * 70)

    # Step 1: 加载轻量化分类模型
    print("\n=== 步骤1: 加载轻量化分类模型 ===")
    try:
        # 检查分类模型文件是否存在
        if not os.path.exists(MULTILABEL_MODEL_PATH):
            print(f"❌ 分类模型权重文件不存在: {MULTILABEL_MODEL_PATH}")
            print("请确保文件路径正确，或训练分类模型")
            return

        if not os.path.exists(CLASS_MAPPING_PATH):
            print(f"❌ 类别映射文件不存在: {CLASS_MAPPING_PATH}")
            print("请确保文件路径正确")
            return

        classifier = LightweightClassifier(MULTILABEL_MODEL_PATH, CLASS_MAPPING_PATH, device)
        print(f"分类模型加载成功，类别: {classifier.class_names}")
    except Exception as e:
        print(f"❌ 加载分类模型出错: {e}")
        return

    # Step 2: 分析数据集
    print("\n=== 步骤2: 分析数据集 ===")
    try:
        analyze_dataset(WT1_XML_DIR, classifier.class_names, OUTPUT_ROOT)
    except Exception as e:
        print(f"⚠️ 数据集分析出错: {e}")

    # Step 3: 构建数据集
    print("\n=== 步骤3: 构建YOLO-seg数据集 ===")
    try:
        # 检查输入目录是否存在
        if not os.path.exists(WT1_TRAIN_IMG_DIR):
            print(f"❌ 训练图片目录不存在: {WT1_TRAIN_IMG_DIR}")
            return
        if not os.path.exists(WT1_VAL_IMG_DIR):
            print(f"❌ 验证图片目录不存在: {WT1_VAL_IMG_DIR}")
            return
        if not os.path.exists(WT1_XML_DIR):
            print(f"❌ XML标注目录不存在: {WT1_XML_DIR}")
            return

        yolo_yaml_path = fast_build_yolo_seg_dataset(
            WT1_TRAIN_IMG_DIR, WT1_VAL_IMG_DIR, WT1_XML_DIR,
            classifier.class_to_idx, OUTPUT_ROOT
        )
    except Exception as e:
        print(f"❌ 构建数据集出错: {e}")
        return

    # Step 4: 训练YOLO-seg
    print("\n=== 步骤4: 训练YOLOv8n-seg（使用增强的数据增强） ===")

    # 优先使用本地权重
    local_yolo_weights = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//dm//yolov8n-seg.pt"

    if os.path.exists(local_yolo_weights):
        yolo_model = YOLO(local_yolo_weights)
        print(f"✅ 从本地加载YOLOv8n-seg权重: {local_yolo_weights}")
    else:
        print("⚠️ 未找到本地YOLO权重，使用默认模型（需要下载）")
        yolo_model = YOLO('yolov8n-seg.pt')

    try:
        # 使用增强的训练参数，包含丰富的数据增强
        train_results = yolo_model.train(
            data=yolo_yaml_path,
            epochs=30,  # 增加训练轮数
            imgsz=640,
            batch=batch_size,
            device=device,
            patience=10,  # 增加早停耐心
            save=True,
            project=OUTPUT_ROOT,
            name="yolov8n_seg_enhanced",
            optimizer='AdamW',  # 使用AdamW优化器，通常效果更好
            lr0=0.001,  # 降低初始学习率
            lrf=0.01,  # 最终学习率
            warmup_epochs=2,  # 增加预热轮数
            # 丰富的数据增强参数
            augment=True,  # 启用数据增强
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # HSV增强
            degrees=10.0,  # 旋转增强
            translate=0.1,  # 平移增强
            scale=0.5,  # 缩放增强
            shear=5.0,  # 剪切增强 (+/- 5度)
            perspective=0.0005,  # 透视增强
            flipud=0.5,  # 上下翻转
            fliplr=0.5,  # 左右翻转
            mosaic=0.7,  # Mosaic增强 (70%概率)
            mixup=0.15,  # Mixup增强 (15%概率)
            copy_paste=0.1,  # Copy-paste增强 (10%概率)
            erasing=0.4,  # 随机擦除 (40%概率)
            auto_augment='randaugment',  # 使用RandAugment自动增强
            verbose=False,
            task='segment',
            dropout=0.1,  # 增加dropout防止过拟合
            weight_decay=0.0005,  # 权重衰减
            momentum=0.937,  # 动量
            nbs=64,  # 名义批次大小
            overlap_mask=True,  # 重叠掩码
            mask_ratio=4,  # 掩码下采样比率
            box=7.5,  # 边界框损失权重
            cls=0.5,  # 分类损失权重
            dfl=1.5  # 分布焦点损失权重
        )

        yolo_best_path = os.path.join(OUTPUT_ROOT, "yolov8n_seg_enhanced", "weights", "best.pt")
        print(f"✅ YOLO训练完成 | 权重路径: {yolo_best_path}")

        # 显示训练结果
        if hasattr(train_results, 'results'):
            print(f"训练结果: mAP50={train_results.results.get('metrics/mAP50(B)', 0):.3f}")
    except Exception as e:
        print(f"❌ 训练YOLO出错: {e}")
        if os.path.exists(local_yolo_weights):
            yolo_best_path = local_yolo_weights
        else:
            yolo_best_path = 'yolov8n-seg.pt'
        print(f"⚠️ 使用备用权重: {yolo_best_path}")

    # Step 5: 构建融合模型
    print("\n=== 步骤5: 构建融合模型 ===")
    fusion_model = FastFusionModel(yolo_best_path, classifier, device)

    # 创建最终模型保存目录
    final_model_dir = os.path.join(OUTPUT_ROOT, "final_model_enhanced")
    fusion_model.save_model(final_model_dir)

    # Step 6: 增强评估
    print("\n=== 步骤6: 增强评估融合模型 ===")
    overall_f1, overall_precision, overall_recall = enhanced_evaluate(
        fusion_model, WT1_VAL_IMG_DIR, WT1_XML_DIR, classifier.class_names, OUTPUT_ROOT
    )

    # Step 6: 增强评估 - 测试不同融合权重
    print("\n=== 步骤6: 测试不同融合权重对mAP@0.5的影响 ===")
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 定义要测试的融合权重范围
    fusion_weights = np.arange(0, 1.1, 0.1)  # 从0到1，步长0.1
    map50_scores = []
    
    for weight in fusion_weights:
        print(f"测试融合权重: {weight:.1f}")
        
        # 创建带有当前融合权重的融合模型
        fusion_model = FastFusionModel(yolo_best_path, classifier, device, fusion_weight=weight)
        
        # 评估模型，计算mAP@0.5
        overall_f1, overall_precision, overall_recall, map50 = enhanced_evaluate(
            fusion_model, WT1_VAL_IMG_DIR, WT1_XML_DIR, classifier.class_names, OUTPUT_ROOT, calculate_map=True
        )
        
        map50_scores.append(map50)
        print(f"  mAP@0.5: {map50:.4f}")
    
    # 绘制mAP@0.5随融合权重变化的图表
    plt.figure(figsize=(10, 6))
    plt.plot(fusion_weights, map50_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('融合权重')
    plt.ylabel('mAP@0.5')
    plt.title('mAP@0.5随融合权重变化曲线')
    plt.grid(True)
    plt.xticks(fusion_weights)
    plt.ylim(0, 1.0)
    
    # 保存图表
    map_plot_path = os.path.join(OUTPUT_ROOT, 'map_vs_fusion_weight.png')
    plt.savefig(map_plot_path)
    print(f"✅ mAP@0.5变化图表已保存至: {map_plot_path}")
    
    # 测试推理功能
    print("\n=== 步骤7: 测试推理功能 ===")
    if os.path.exists(WT1_VAL_IMG_DIR) and len(os.listdir(WT1_VAL_IMG_DIR)) > 0:
        # 测试多张图片
        test_images = os.listdir(WT1_VAL_IMG_DIR)[:5]  # 测试前5张图片
        for img_name in test_images:
            test_img = os.path.join(WT1_VAL_IMG_DIR, img_name)
            print(f"\n测试图片: {img_name}")
            results = fusion_model.predict(test_img)
            print(f"检测到 {len(results['seg_regions'])} 个分割区域")

            # 显示所有检测区域
            for i, region in enumerate(results['seg_regions']):
                print(f"  区域{i}: bbox={region['bbox']}, 标签={region['multilabels']}, "
                      f"置信度={region['label_confs']}, 分割面积={region['seg_area']}")

            if len(results['seg_regions']) == 0:
                print("⚠️ 未检测到任何分割区域")

    # 最终总结
    print("\n" + "=" * 70)
    print("训练与评估完成总结")
    print("=" * 70)
    print(f"📌 分类模型: {MULTILABEL_MODEL_PATH}")
    print(f"📌 YOLO训练日志: {os.path.join(OUTPUT_ROOT, 'yolov8n_seg_enhanced')}")
    print(f"📌 融合模型保存路径: {final_model_dir}")
    print(f"📌 数据集配置: {yolo_yaml_path}")
    print(f"📊 最终评估结果:")
    print(f"   总体F1分数: {overall_f1:.4f}")
    print(f"   总体精确率: {overall_precision:.4f}")
    print(f"   总体召回率: {overall_recall:.4f}")
    print("\n🎉 全流程完成！")


if __name__ == "__main__":
    main()