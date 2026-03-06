import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import warnings
from collections import Counter
import pandas as pd

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_VISUALIZATION_LIBS = True
except ImportError:
    HAS_VISUALIZATION_LIBS = False
    warnings.warn("未安装 matplotlib 或 seaborn，将跳过混淆矩阵可视化。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 路径定义
train_img_dir = r"//mmbdsj//code//fs//sj//final_train"
val_img_dir = r"//mmbdsj//code//fs//sj//final_val"
test_img_dir = r"//mmbdsj//code//fs//sj//final_test"
annotations_dir = r"//mmbdsj//code//fs//sj//Annotations"
output_dir = r"//mmbdsj//code//fs//sj//wt1_zz"
# 本地ResNet50预训练权重路径
local_weight_path = r"//mmbdsj//code//fs//dm//resnet50-0676ba61.pth"
os.makedirs(output_dir, exist_ok=True)


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
    def __init__(self, img_dir, annotations, class_to_idx, transform=None):
        self.img_dir = img_dir
        self.annotations = annotations
        self.class_to_idx = class_to_idx
        self.transform = transform
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
        print(f"加载 {len(self.img_names)} 张图像用于多标签训练")

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


# 新增：无标注的测试集数据集
class UnlabeledDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = []
        supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']
        for img_name in os.listdir(img_dir):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in supported_ext:
                self.img_names.append(img_name)
        print(f"找到 {len(self.img_names)} 张图像用于批量预测")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        if self.transform:
            image = self.transform(image)
        return image, img_name


# ========================
# 创建多标签模型（ResNet50）-
# ========================
def create_multilabel_model(num_classes):
    # 1. 初始化空的ResNet50模型
    model = models.resnet50(pretrained=False)
    # 2. 加载本地的ResNet50预训练权重文件
    if os.path.exists(local_weight_path):
        print(f"加载本地预训练权重：{local_weight_path}")
        state_dict = torch.load(local_weight_path, map_location=device)
        # 加载权重到模型
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        raise FileNotFoundError(f"本地权重文件不存在：{local_weight_path}")

    # 3. 冻结部分参数，微调指定层
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 4. 替换全连接层为多标签分类头
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# 新增：加载模型工具函数
def load_trained_model(model_path, class_mapping_path):
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    class_to_idx = mappings['class_to_idx']
    num_classes = len(class_to_idx)

    model = create_multilabel_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, mappings


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
# 混淆矩阵相关
# ========================
def save_and_plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    if not HAS_VISUALIZATION_LIBS:
        print("跳过混淆矩阵可视化（缺少 matplotlib/seaborn）")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

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

        # 归一化按行
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        # 保存 CSV
        df_raw = pd.DataFrame(cm,
                              index=['实际负', '实际正'],
                              columns=['预测负', '预测正'])
        df_norm = pd.DataFrame(cm_norm,
                               index=['实际负', '实际正'],
                               columns=['预测负', '预测正'])
        df_raw.to_csv(os.path.join(cm_csv_dir, f'cm_{cls}_raw.csv'), encoding='utf-8-sig')
        df_norm.to_csv(os.path.join(cm_csv_dir, f'cm_{cls}_normalized.csv'), encoding='utf-8-sig')

        # 绘制原始
        ax_raw = axes_raw[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_raw,
                    xticklabels=['预测负', '预测正'],
                    yticklabels=['实际负', '实际正'])
        ax_raw.set_title(f'类别: {cls}')

        # 绘制归一化
        ax_norm = axes_norm[i]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax_norm,
                    xticklabels=['预测负', '预测正'],
                    yticklabels=['实际负', '实际正'])
        ax_norm.set_title(f'类别: {cls} (归一化)')

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

        class_names = [idx_to_class[i] for i in range(len(all_true[0]))]
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


# 新增：批量预测函数
def batch_predict(model, img_dir, class_to_idx, idx_to_class, transform, batch_size=16, threshold=0.5):
    """
    对文件夹中的所有图像进行批量多标签预测
    """
    dataset = UnlabeledDataset(img_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 4),
        pin_memory=True
    )

    results = {}
    model.eval()
    with torch.no_grad():
        for inputs, img_names in tqdm(dataloader, desc='批量预测'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            for i, img_name in enumerate(img_names):
                pred_labels = []
                pred_probs = {}
                for cls_name, idx in class_to_idx.items():
                    if probs[i, idx] > threshold:
                        pred_labels.append(cls_name)
                        pred_probs[cls_name] = float(probs[i, idx])

                # 处理无预测结果的情况
                if not pred_labels:
                    pred_labels = ["normal"]

                results[img_name] = {
                    'labels': pred_labels,
                    'probabilities': pred_probs
                }

    return results


# 新增：保存预测结果
def save_prediction_results(results, output_path):
    """保存批量预测结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 同时保存为CSV格式，方便查看
    csv_data = []
    for img_name, data in results.items():
        csv_data.append({
            'image_name': img_name,
            'predicted_labels': ','.join(data['labels']),
            'probabilities': json.dumps(data['probabilities'], ensure_ascii=False)
        })

    df = pd.DataFrame(csv_data)
    csv_path = os.path.splitext(output_path)[0] + '.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return csv_path


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    try:
        print("=== 加载多标签标注 ===")
        annotations, class_to_idx = get_all_annotations_multi(annotations_dir)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        class_names = [idx_to_class[i] for i in range(num_classes)]
        print(f"类别列表: {class_names}")

        # 数据变换
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 预测用的变换（与验证集一致）
        predict_transform = val_transform

        print("=== 创建数据集 ===")
        train_dataset = MultiLabelContainerDataset(train_img_dir, annotations, class_to_idx, transform=train_transform)
        val_dataset = MultiLabelContainerDataset(val_img_dir, annotations, class_to_idx, transform=val_transform)

        batch_size = 16
        num_workers = min(4, os.cpu_count() or 4)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True)

        print("=== 初始化模型 ===")
        model = create_multilabel_model(num_classes)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

        model_save_path = os.path.join(output_dir, 'best_multilabel_model.pth')
        print("=== 开始训练 ===")
        history, best_f1 = train_multilabel_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=25, model_save_path=model_save_path, patience=5
        )

        # 保存训练历史
        history_path = os.path.join(output_dir, 'multilabel_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print("\n=== 评估最佳模型 ===")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        all_true, all_pred = [], []
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc='评估'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_true.extend(labels.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())

        # 保存评估结果
        eval_results_path = os.path.join(output_dir, 'evaluation_results.npz')
        np.savez(eval_results_path, y_true=np.array(all_true), y_pred=np.array(all_pred))
        print(f"✅ 评估结果已保存至: {eval_results_path}")

        # 生成并保存评估报告
        report = multilabel_classification_report(all_true, all_pred, class_names)
        print_multilabel_report(report, class_names)

        report_txt_path = os.path.join(output_dir, 'multilabel_classification_report.txt')
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write("多标签分类报告\n")
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

        # 保存最终模型和类别映射
        final_model_path = os.path.join(output_dir, 'final_multilabel_model.pth')
        torch.save(model.state_dict(), final_model_path)
        mapping_path = os.path.join(output_dir, 'multilabel_class_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, f, indent=4, ensure_ascii=False)

        # 保存混淆矩阵
        save_and_plot_confusion_matrices(np.array(all_true), np.array(all_pred), class_names, output_dir)

        # 新增：批量预测测试集
        print("\n=== 开始批量预测 ===")
        if os.path.exists(test_img_dir) and len(os.listdir(test_img_dir)) > 0:
            # 加载最佳模型进行预测
            model, mappings = load_trained_model(model_save_path, mapping_path)

            # 执行批量预测
            predictions = batch_predict(
                model,
                test_img_dir,
                mappings['class_to_idx'],
                mappings['idx_to_class'],
                predict_transform,
                batch_size=batch_size,
                threshold=0.5  # 可根据需求调整阈值
            )

            # 保存预测结果
            pred_output_path = os.path.join(output_dir, 'batch_predictions.json')
            csv_path = save_prediction_results(predictions, pred_output_path)
            print(f"✅ 批量预测结果已保存至: {pred_output_path} 和 {csv_path}")
        else:
            print("⚠️ 未找到测试集图像，跳过批量预测")

        print(f"\n✅ 所有输出已成功保存至: {output_dir}")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()