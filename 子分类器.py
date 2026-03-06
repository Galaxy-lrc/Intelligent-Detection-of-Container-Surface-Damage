import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
import glob
from collections import defaultdict

# ===================== 1. 全局配置与路径定义 =====================
# 设备配置：优先使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


TRAIN_ROOT = r"//mmbdsj//code//fs//sj//final_train"  # 训练集图像目录
VAL_ROOT = r"//mmbdsj//code//fs//sj//final_val"    # 验证集图像目录
ANNOTATION_ROOT = r"//mmbdsj//code//fs//sj//Annotations"  # 所有XML标注目录

# 级联分类器对应的类别
CLASSIFIER_A_CLASSES = ["scratch", "broken", "rusty"]  # 子分类器A类别
CLASSIFIER_B_CLASSES = ["broken", "hole"]  # 子分类器B类别

# 统一保存文件夹（所有模型、曲线都存放在这里）
SAVE_DIR = "//mmbdsj//code//fs//jlfl"
# 创建保存文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 2. 数据加载与预处理 =====================
class XMLCustomDataset(Dataset):
    """
    自定义数据集类：
    1. 解析XML标注文件（按文件名匹配图像，且校验图像所属目录）
    2. 严格隔离训练/验证集：仅保留当前目录下的图像对应的标注，过滤跨目录样本
    3. 筛选指定类别样本用于级联分类器训练
    """
    def __init__(self, image_dir, annotation_dir, classes, is_train=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes if classes is not None else []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.is_train = is_train
        # 加载当前目录下的样本
        self.samples = self._load_samples()
        # 检查样本数量
        if len(self.samples) == 0:
            raise ValueError(
                f"❌ 未找到指定类别{self.classes}的样本，请检查：\n"
                f"1. XML中的类别名是否与代码中的匹配\n"
                f"2. 图像与XML文件是否按文件名对应\n"
                f"3. 图像目录{self.image_dir}是否包含对应文件\n"
                f"4. XML标注是否对应当前目录的图像"
            )
        print(f"✅ 成功加载{len(self.samples)}个{self.classes}类别的样本（{self.image_dir}）")
        self.transform = self._get_transform()

    def _get_image_files(self):
        """获取当前图像目录下的所有图像文件"""
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
        image_files = []
        for ext in image_extensions:
            # 递归查找
            image_files.extend(glob.glob(os.path.join(self.image_dir, f"**/*{ext}"), recursive=True))
            image_files.extend(glob.glob(os.path.join(self.image_dir, f"**/*{ext.upper()}"), recursive=True))
        # 去重
        image_files = list(set(image_files))
        if not image_files:
            raise ValueError(f"❌ 图像目录{self.image_dir}中未找到图像文件")
        return image_files

    def _parse_xml_label(self, xml_path):
        """解析XML文件，提取类别标签"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            label = None
            for obj in root.findall("object"):
                label = obj.find("name").text
                break
            return label
        except Exception as e:
            print(f"⚠️ 解析XML文件{xml_path}失败，错误：{e}")
            return None

    def _verify_image_in_dir(self, img_path, xml_img_name):
        """验证XML对应的图像是否确实在当前目录下"""
        # 查找当前目录下是否有该名称的图像
        matching_imgs = glob.glob(os.path.join(self.image_dir, f"**/{xml_img_name}*"), recursive=True)
        return len(matching_imgs) > 0

    def _load_samples(self):
        """加载样本：严格匹配当前目录的图像+对应XML标注，过滤跨目录样本"""
        samples = []
        # 1. 获取当前目录的所有图像文件
        image_files = self._get_image_files()
        # 2. 建立文件名到图像路径的映射
        img_name_to_path = defaultdict(list)
        for img_path in image_files:
            img_basename = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_basename)[0]
            img_name_to_path[img_name_no_ext].append(img_path)

        # 3. 遍历所有XML标注文件，仅匹配当前目录的图像
        xml_files = glob.glob(os.path.join(self.annotation_dir, "*.xml"))
        for xml_path in xml_files:
            xml_basename = os.path.basename(xml_path)
            xml_name_no_ext = os.path.splitext(xml_basename)[0]
            # 检查当前目录是否有该XML对应的图像
            if xml_name_no_ext not in img_name_to_path:
                continue

            # 4. 解析XML获取类别标签
            label = self._parse_xml_label(xml_path)
            if label is None or label not in self.classes:
                continue

            # 5. 遍历匹配的图像路径，添加样本
            for img_path in img_name_to_path[xml_name_no_ext]:
                samples.append((img_path, self.class_to_idx[label]))

        return samples

    def _get_transform(self):
        """数据预处理：训练集增强，验证集仅归一化"""
        if self.is_train:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        return transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ 跳过损坏的图像文件：{img_path}，错误：{e}")
            # 临时生成空白图像避免训练中断
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        img = self.transform(img)
        return img, label

def get_cascade_dataloader(image_dir, annotation_dir, classes, batch_size=16, is_train=True):
    """
    获取级联分类器的DataLoader
    :param image_dir: 训练/验证图像文件夹路径
    :param annotation_dir: XML标注文件夹路径
    :param classes: 待分类的类别列表
    :param batch_size: 批次大小
    :param is_train: 是否为训练集
    :return: DataLoader, class_to_idx
    """
    dataset = XMLCustomDataset(image_dir, annotation_dir, classes, is_train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader, dataset.class_to_idx

# ===================== 3. 模型构建 =====================
def build_resnet50_model(num_classes):
    """
    构建ResNet50模型
    :param num_classes: 类别数（子分类器A：3；子分类器B：2）
    :return: 模型
    """
    # 加载预训练ResNet50
    try:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except ImportError:
        model = models.resnet50(pretrained=True)

    # 微调策略：解冻后两层和全连接层
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 替换最后一层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # 移至设备
    model = model.to(device)
    return model

# ===================== 4. 训练与验证函数 =====================
def train_cascade_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, model_name="model"):
    """
    训练级联分类器模型
    :param model: 模型
    :param train_loader: 训练集加载器
    :param val_loader: 验证集加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param num_epochs: 训练轮数
    :param model_name: 模型保存名称前缀
    :return: 训练好的模型
    """
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    lr_history = []

    # 定义模型保存路径
    best_model_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")
    final_model_path = os.path.join(SAVE_DIR, f"{model_name}_final.pth")
    curve_path = os.path.join(SAVE_DIR, f"{model_name}_train_curve.png")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # 计算训练损失
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        lr_history.append(optimizer.param_groups[0]['lr'])

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        epoch_acc = correct / total
        val_accs.append(epoch_acc)
        print(f"\n{model_name} - Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Val Acc = {epoch_acc:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # 更新学习率
        scheduler.step(epoch_acc)

        # 保存最优模型
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最优{model_name}模型至：{best_model_path}，验证集准确率：{best_val_acc:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label=f"{model_name} Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(val_accs, label=f"{model_name} Val Acc", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    # 学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(lr_history, label=f"{model_name} LR", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.title("Learning Rate History")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.show()
    print(f"训练曲线保存至：{curve_path}")

    # 加载最优模型，再保存最终模型
    model.load_state_dict(torch.load(best_model_path))
    torch.save(model.state_dict(), final_model_path)
    print(f"最终{model_name}模型保存至：{final_model_path}")

    return model

# ===================== 5. XML标注调试函数 =====================
def debug_xml_annotation():
    """调试XML标注文件，查看文件名、类别名及对应图像是否存在于训练/验证目录"""
    xml_files = glob.glob(os.path.join(ANNOTATION_ROOT, "*.xml"))
    if not xml_files:
        print("❌ 标注文件夹中没有XML文件！")
        return

    # 检查训练/验证目录的图像匹配情况
    train_img_names = set()
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
        train_img_names.update([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(TRAIN_ROOT, f"**/*{ext}"), recursive=True)])
        train_img_names.update([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(TRAIN_ROOT, f"**/*{ext.upper()}"), recursive=True)])

    val_img_names = set()
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
        val_img_names.update([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(VAL_ROOT, f"**/*{ext}"), recursive=True)])
        val_img_names.update([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(VAL_ROOT, f"**/*{ext.upper()}"), recursive=True)])

    # 打印前5个XML文件的信息
    for i, xml_path in enumerate(xml_files[:5]):
        xml_basename = os.path.basename(xml_path)
        xml_name_no_ext = os.path.splitext(xml_basename)[0]
        # 解析XML标签
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            label = None
            for obj in root.findall("object"):
                label = obj.find("name").text
                break
        except Exception as e:
            label = f"解析失败：{e}"

        # 检查所属目录
        in_train = xml_name_no_ext in train_img_names
        in_val = xml_name_no_ext in val_img_names
        print(f"\n===== XML文件 {i+1}：{xml_basename} =====")
        print(f"提取的类别标签：{label}")
        print(f"是否在训练集目录：{in_train}")
        print(f"是否在验证集目录：{in_val}")
        if in_train and in_val:
            print(f"⚠️ 该样本同时出现在训练集和验证集目录，存在泄露风险！")

# ===================== 6. 主函数：训练级联分类器 =====================
if __name__ == "__main__":
    # 可选：运行调试函数，查看XML标注的类别标签和样本分布
    debug_xml_annotation()

    # 打印保存路径信息
    print(f"\n所有模型和训练曲线将保存至：{os.path.abspath(SAVE_DIR)}")

    # -------------------------- 训练子分类器A（scratch/broken/rusty） --------------------------
    print("\n========== 开始训练子分类器A（区分scratch/broken/rusty）==========")
    # 加载训练集和验证集
    train_loader_A, class_to_idx_A = get_cascade_dataloader(
        image_dir=TRAIN_ROOT,
        annotation_dir=ANNOTATION_ROOT,
        classes=CLASSIFIER_A_CLASSES,
        batch_size=16,
        is_train=True
    )
    val_loader_A, _ = get_cascade_dataloader(
        image_dir=VAL_ROOT,
        annotation_dir=ANNOTATION_ROOT,
        classes=CLASSIFIER_A_CLASSES,
        batch_size=16,
        is_train=False
    )
    print(f"子分类器A类别映射：{class_to_idx_A}")

    # 构建模型（3个类别）
    model_A = build_resnet50_model(num_classes=len(CLASSIFIER_A_CLASSES))

    # 定义损失函数、优化器（学习率0.005）、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer_A = optim.Adam(model_A.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler_A = ReduceLROnPlateau(
        optimizer_A,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # 训练模型
    model_A = train_cascade_model(
        model=model_A,
        train_loader=train_loader_A,
        val_loader=val_loader_A,
        criterion=criterion,
        optimizer=optimizer_A,
        scheduler=scheduler_A,
        num_epochs=20,
        model_name="subclassifier_A"
    )
    print("子分类器A训练完成！")

    # -------------------------- 训练子分类器B（broken/hole） --------------------------
    print("\n========== 开始训练子分类器B（区分broken/hole）==========")
    # 加载训练集和验证集（严格隔离，无泄露）
    train_loader_B, class_to_idx_B = get_cascade_dataloader(
        image_dir=TRAIN_ROOT,
        annotation_dir=ANNOTATION_ROOT,
        classes=CLASSIFIER_B_CLASSES,
        batch_size=16,
        is_train=True
    )
    val_loader_B, _ = get_cascade_dataloader(
        image_dir=VAL_ROOT,
        annotation_dir=ANNOTATION_ROOT,
        classes=CLASSIFIER_B_CLASSES,
        batch_size=16,
        is_train=False
    )
    print(f"子分类器B类别映射：{class_to_idx_B}")

    # 构建模型（2个类别）
    model_B = build_resnet50_model(num_classes=len(CLASSIFIER_B_CLASSES))

    # 定义损失函数、优化器、调度器
    optimizer_B = optim.Adam(model_B.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler_B = ReduceLROnPlateau(
        optimizer_B,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # 训练模型
    model_B = train_cascade_model(
        model=model_B,
        train_loader=train_loader_B,
        val_loader=val_loader_B,
        criterion=criterion,
        optimizer=optimizer_B,
        scheduler=scheduler_B,
        num_epochs=20,
        model_name="subclassifier_B"
    )
    print("子分类器B训练完成！")
    print(f"\n所有文件已保存至：{os.path.abspath(SAVE_DIR)}")