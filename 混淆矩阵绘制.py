import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========================
# 路径配置（请确认路径正确性）
# ========================
test_img_dir = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//final_val"  # 测试集路径
annotations_dir = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//Annotations"  # 标签路径
model_path = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//dm//best_multilabel_model_scene_aware_reflection.pth"  # 训练好的模型路径
class_mapping_path = r"D://数学建模//25妈妈杯大数据//复赛//25妈妈杯大数据复赛//数据//multilabel_class_mapping.json"  # 类别映射文件路径（训练时生成）


# ========================
# 核心工具函数
# ========================
def parse_xml_annotation_multi(xml_file):
    """解析XML获取多标签"""
    labels = set()
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None and name_elem.text is not None:
                labels.add(name_elem.text.strip())
    except Exception as e:
        print(f"解析XML {xml_file} 出错: {e}")
    return list(labels) if labels else ["normal"]


def get_all_annotations_multi(annotations_dir):
    """获取所有标注和类别映射"""
    annotations = {}
    all_classes = set()
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    for xml_file in xml_files:
        xml_path = os.path.join(annotations_dir, xml_file)
        img_name = os.path.splitext(xml_file)[0] + '.jpg'
        labels = parse_xml_annotation_multi(xml_path)
        annotations[img_name] = labels
        all_classes.update(labels)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    return annotations, class_to_idx


# ========================
# 数据集类（仅加载测试集+标签）
# ========================
class SimpleMultiLabelDataset(Dataset):
    def __init__(self, img_dir, annotations, class_to_idx):
        self.img_dir = img_dir
        self.annotations = annotations
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose([  # 与训练时一致的预处理
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 筛选有效图像
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg') and f in annotations]
        print(f"加载测试图像数量: {len(self.img_names)}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # 加载图像
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        image = self.transform(image)

        # 加载真实标签向量
        labels = self.annotations[img_name]
        label_vec = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                label_vec[self.class_to_idx[label]] = 1.0
        return image, label_vec


# ========================
# 模型加载（与训练时结构一致）
# ========================
def load_multilabel_model(num_classes, model_path):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # 替换最后一层（与训练时一致）
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# ========================
# 核心：绘制目标样式的归一化混淆矩阵
# ========================
def plot_target_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制与示例一致的归一化混淆矩阵"""
    # 移除中文字体设置
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 计算混淆矩阵（此处代码不变）
    cm = np.zeros((len(class_names), len(class_names)))
    for t, p in zip(y_true, y_pred):
        t_idx = np.where(t == 1)[0] if len(np.where(t == 1)[0]) > 0 else [0]
        p_idx = np.where(p == 1)[0] if len(np.where(p == 1)[0]) > 0 else [0]
        # 多标签转单标签（适配混淆矩阵可视化，若需纯多标签可调整）
        t_main = t_idx[0] if len(t_idx) > 0 else 0
        p_main = p_idx[0] if len(p_idx) > 0 else 0
        cm[t_main, p_main] += 1

    # 归一化（按行）（此处代码不变）
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    cm_normalized = np.nan_to_num(cm_normalized)

    # 绘制核心混淆矩阵（修改中文为英文）
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,  # 显示数值
        fmt='.2f',  # 保留2位小数
        cmap='Blues',  # 蓝色系（与示例一致）
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        cbar_kws={"label": "Normalized Value"}  # 颜色条标签改为英文
    )
    plt.xlabel('Predicted Label')  # x轴标签改为英文
    plt.ylabel('True Label')      # y轴标签改为英文
    plt.title('Normalized Confusion Matrix')  # 标题改为英文
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 混淆矩阵已保存至: {save_path}")  # 控制台输出保持中文（根据需求仅修改可视化部分）


# ========================
# 主流程：推理+可视化
# ========================
if __name__ == "__main__":
    # 1. 加载标注和类别映射
    annotations, class_to_idx = get_all_annotations_multi(annotations_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    num_classes = len(class_to_idx)
    print(f"类别列表: {class_names}")

    # 2. 加载数据集
    test_dataset = SimpleMultiLabelDataset(test_img_dir, annotations, class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 3. 加载模型
    model = load_multilabel_model(num_classes, model_path)

    # 4. 推理获取真实标签和预测标签
    all_true = []
    all_pred = []
    print("\n开始推理测试集...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 多标签阈值判断（与训练一致）
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_true.extend(labels.numpy())
            all_pred.extend(preds)

    # 5. 绘制并保存目标混淆矩阵
    save_path = os.path.join(os.path.dirname(model_path), "confusion_matrix_normalized.png")
    plot_target_confusion_matrix(all_true, all_pred, class_names, save_path)