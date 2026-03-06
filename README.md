# Intelligent-Detection-of-Container-Surface-Damage
Intelligent Detection of Container Surface Damage Based on Scene Self-Adaptation and Cascaded Network
# Intelligent Road Defect Detection System Based on Cascade Classification and Scene Awareness

## 📝 Project Overview

This project proposes an intelligent road defect detection system based on cascade classification and scene-aware multi-task learning. Through a multi-level classification architecture, scene-adaptive preprocessing, and lightweight model design, the system achieves high-precision recognition and localization of various road defects (such as cracks, breakage, rust, holes, etc.) in complex traffic scenarios.

### Key Features

- **Cascade Classification Architecture**: Adopts a two-level cascade classification strategy for coarse-to-fine defect recognition
- **Scene-Adaptive Preprocessing**: Develops specialized preprocessing modules for complex scenarios like rainy day reflections
- **Lightweight Model Design**: ResNet50-based lightweight multi-label classification model balancing accuracy and efficiency
- **End-to-End Pipeline**: Supports complete processing workflow from raw images to defect annotations

## 🚀 Technical Highlights

### 1. Cascade Classification System

**First-Level Classifier**:
- Responsible for coarse-grained defect classification
- Categories: scratch, broken, rusty, hole

**Second-Level Classifier**:
- Sub-classifier A: Fine-grained classification for scratch, broken, rusty
- Sub-classifier B: Fine-grained classification for broken, hole

### 2. Scene-Aware Preprocessing

- **Ground Reflection Detection**: Based on brightness threshold and saturation analysis
- **Adaptive Image Enhancement**: CLAHE brightness equalization, HSV color adjustment
- **Multi-scale Denoising**: Median filtering, edge enhancement

### 3. Multi-Label Classification Model

- Supports multiple defect labels in a single image
- ResNet50 backbone network, freeze first two layers and fine-tune last two layers
- Sigmoid activation function for multi-label prediction

## 📁 Project Structure

```
.
├── 1.py                    # Lightweight multi-label classification model (main model)
├── 级联分类.py              # Cascade classifier training and inference
├── 子分类器.py              # Sub-classifier training script
├── 问题一分场训练.py        # Scene-aware training module
├── 预处理对比图.py          # Preprocessing effect visualization
├── data/                   # Data directory (prepare yourself)
│   ├── final_train/        # Training set images
│   ├── final_val/          # Validation set images
│   ├── final_test/         # Test set images
│   └── Annotations/        # XML annotation files
├── models/                 # Model weights directory
└── outputs/                # Output results directory
```

## 🛠️ Environment Setup

### System Requirements
- Python >= 3.8
- CUDA >= 11.0 (GPU recommended)
- RAM >= 16GB

### Installation

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install pillow
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install pyyaml
pip install scikit-learn
pip install tqdm
```

## 📖 Data Preparation

### Dataset Format

```
data/
├── final_train/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── final_val/
│   ├── image003.jpg
│   └── ...
└── Annotations/
    ├── image001.xml
    ├── image002.xml
    └── ...
```

### XML Annotation Format

```xml
<annotation>
    <filename>image001.jpg</filename>
    <size>
        <width>1920</width>
        <height>1080</height>
    </size>
    <object>
        <name>scratch</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>
```

## 💻 Usage

### 1. Train Main Model (Multi-label Classification)

```bash
python 1.py
```

**Key Parameters**:
- `train_img_dir`: Training set image path
- `val_img_dir`: Validation set image path
- `annotations_dir`: XML annotation path
- `output_dir`: Output directory
- `batch_size`: Batch size (default: 16)

### 2. Train Cascade Classifier

```bash
python 级联分类.py
```

**Cascade Classifier Categories**:
- Sub-classifier A: `["scratch", "broken", "rusty"]`
- Sub-classifier B: `["broken", "hole"]`

### 3. Train Sub-classifier

```bash
python 子分类器.py
```

### 4. Scene-Aware Training

```bash
python 问题一分场训练.py
```

**Features**:
- Ground reflection detection
- Rainy scene preprocessing
- Scene-specific model training

### 5. Preprocessing Effect Visualization

```bash
python 预处理对比图.py
```

**Input**: Single image path  
**Output**: Comparison of preprocessing stages (including histogram and edge detection results)

## 🎯 Model Inference

### Single Image Inference Example

```python
from 1 import LightweightClassifier

# Load model
model = LightweightClassifier(
    model_path="models/best_model.pth",
    class_mapping_path="models/class_mapping.json",
    device="cuda"
)

# Inference
image = Image.open("test.jpg")
pred_labels, pred_confs = model.predict(image)

print("Prediction Results:")
for label, conf in zip(pred_labels, pred_confs):
    print(f"  - {label}: {conf:.4f}")
```

### Batch Inference

```python
import glob
from PIL import Image

test_images = glob.glob("data/final_test/*.jpg")
for img_path in test_images:
    image = Image.open(img_path)
    pred_labels, pred_confs = model.predict(image)
    print(f"{img_path}: {pred_labels}")
```

## 📊 Performance Evaluation

### Evaluation Metrics

- **Multi-label Classification**: mAP, F1-score, Precision, Recall
- **Confusion Matrix**: Supports multi-label confusion matrix visualization
- **ROC Curve**: ROC curves for each label

### Run Evaluation

```bash
python 1.py --mode evaluate --model_path models/best_model.pth
```

## 🔧 Advanced Configuration

### Model Fine-tuning

Modify the following parameters in `1.py`:

```python
# Freezing strategy
for param in model.parameters():
    param.requires_grad = False

# Fine-tune last two layers
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
```

### Data Augmentation

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

## 📈 Expected Performance

| Scenario | Metric | Main Model | Cascade Classifier |
|----------|--------|------------|-------------------|
| Normal Scenario | mAP | 0.92 | 0.95 |
| Rainy Scenario | mAP | 0.85 | 0.90 |
| Low-light Scenario | mAP | 0.88 | 0.92 |

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

If you have any questions, please contact us through Issues.

## 🙏 Acknowledgments

- PyTorch
- Ultralytics YOLO
- torchvision

---

**Note**: Please modify the data paths, model paths, and other configuration parameters according to your actual requirements.
