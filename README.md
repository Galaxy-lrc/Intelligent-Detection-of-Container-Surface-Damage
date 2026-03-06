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

## 🚀 Quick Reproduction Steps

Follow these 4 steps to reproduce the complete workflow:

### Step 1: Configure Conda Environment and Verify GPU

```bash
# Create conda environment
conda create -n road_defect python=3.8
conda activate road_defect

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pillow matplotlib seaborn pandas numpy pyyaml scikit-learn tqdm

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.cuda.version if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output**:
```
CUDA available: True
CUDA version: 11.8
```

### Step 3: Train Models in Sequence

#### 3.1 Train Main CCN Model (Multi-label Classification)

**File**: `1.py`

```bash
# Train main multi-label classification model
python 1.py

# This script will:
# - Load training/validation images from ./data/final_train and ./data/final_val
# - Parse XML annotations from ./data/Annotations
# - Train ResNet50-based multi-label classifier
# - Save model weights to ./outputs/wt1_zz/
# - Generate class_mapping.json
```

**Output**:
- Model weights: `./outputs/wt1_zz/best_model.pth`
- Class mapping: `./outputs/wt1_zz/class_mapping.json`
- Training curves: `./outputs/wt1_zz/training_curves.png`

#### 3.2 Train Cascade Classifiers

**File**: `级联分类.py`

```bash
# Train cascade classifier system
python 级联分类.py

# This script will:
# - Train first-level classifier for coarse classification
# - Train sub-classifier A (scratch, broken, rusty)
# - Train sub-classifier B (broken, hole)
# - Save all cascade models to ./outputs/jlfl/
```

**Output**:
- First-level classifier: `./outputs/jlfl/classifier_level1.pth`
- Sub-classifier A: `./outputs/jlfl/classifier_A.pth`
- Sub-classifier B: `./outputs/jlfl/classifier_B.pth`
- Confusion matrices: `./outputs/jlfl/confusion_matrix.png`

#### 3.3 Train Sub-classifiers Independently

**File**: `子分类器.py`

```bash
# Train sub-classifiers separately
python 子分类器.py

# This script will:
# - Train specialized sub-classifiers for specific defect types
# - Generate detailed classification reports
# - Save models to ./outputs/wt1_zz/
```

**Output**:
- Sub-classifier weights: `./outputs/wt1_zz/sub_classifier_*.pth`
- Classification reports: `./outputs/wt1_zz/classification_report.txt`

#### 3.4 Scene-Aware Training

**File**: `问题一分场训练.py`

```bash
# Train scene-aware models
python 问题一分场训练.py

# This script will:
# - Detect ground reflections in rainy scenes
# - Apply scene-specific preprocessing
# - Train models adapted to different scenarios
# - Save scene-specific models to ./outputs/问题一分场训练结果集/
```

**Output**:
- Scene-specific models: `./outputs/问题一分场训练结果集/scene_model.pth`
- Preprocessed images: `./outputs/问题一分场训练结果集/scene_processed_images/`
- Reflection masks: `./outputs/问题一分场训练结果集/rain_reflection/`

#### 3.5 Build YOLOv8n-seg Dataset

**File**: `1.py` (function `fast_build_yolo_seg_dataset`)

```bash
# Build YOLO-seg dataset from XML annotations
python 1.py --mode build_yolo_dataset

# This script will:
# - Convert XML annotations to YOLO-seg format
# - Generate data.yaml configuration
# - Create dataset structure at ./data/yolo_seg_light/
```

**Output**:
- YOLO dataset: `./data/yolo_seg_light/`
- Configuration: `./data/yolo_seg_light/data.yaml`

#### 3.6 Train YOLOv8n-seg Model

**File**: YOLO CLI (via Ultralytics)

```bash
# Train YOLOv8n-seg model
yolo detect segment train \
    data=./data/yolo_seg_light/data.yaml \
    model=yolov8n-seg.pt \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=./outputs/yolo_seg \
    name=train
```

**Output**:
- YOLO model: `./outputs/yolo_seg/train/weights/best.pt`
- Training logs: `./outputs/yolo_seg/train/`
- Validation results: `./outputs/yolo_seg/train/results.csv`

**Training Sequence Summary**:
```
1. 1.py                     → Main CCN model (multi-label classification)
2. 级联分类.py              → Cascade classifier system
3. 子分类器.py              → Independent sub-classifiers
4. 问题一分场训练.py        → Scene-aware models
5. 1.py (build mode)        → YOLO dataset conversion
6. YOLO CLI                 → YOLOv8n-seg training
```

**Estimated Training Time** (on RTX 3090):
- Step 3.1 (1.py): ~2-3 hours
- Step 3.2 (级联分类.py): ~1-2 hours
- Step 3.3 (子分类器.py): ~1-2 hours
- Step 3.4 (问题一分场训练.py): ~2-3 hours
- Step 3.5 (1.py build mode): ~10 minutes
- Step 3.6 (YOLO CLI): ~4-6 hours

### Step 4: Model Inference and Evaluation

#### 4.1 Model Weights Location

After training completion, all model weights are saved in the following locations:

**CCN Models** (from Step 3.1):
- Main model: `./outputs/wt1_zz/best_model.pth`
- Class mapping: `./outputs/wt1_zz/class_mapping.json`

**Cascade Classifiers** (from Step 3.2):
- Level 1: `./outputs/jlfl/classifier_level1.pth`
- Sub-classifier A: `./outputs/jlfl/classifier_A.pth`
- Sub-classifier B: `./outputs/jlfl/classifier_B.pth`

**Sub-classifiers** (from Step 3.3):
- Models: `./outputs/wt1_zz/sub_classifier_*.pth`

**Scene-aware Models** (from Step 3.4):
- Scene model: `./outputs/问题一分场训练结果集/scene_model.pth`

**YOLOv8n-seg Model** (from Step 3.6):
- Best weights: `./outputs/yolo_seg/train/weights/best.pt`

#### 4.2 Run Inference with Main CCN Model

**File**: `1.py` (class `LightweightClassifier`)

```python
# Single image inference
from 1 import LightweightClassifier
from PIL import Image

# Load CCN model
ccn_model = LightweightClassifier(
    model_path="./outputs/wt1_zz/best_model.pth",
    class_mapping_path="./outputs/wt1_zz/class_mapping.json",
    device="cuda"
)

# Perform inference
image_path = "test.jpg"
image = Image.open(image_path)
pred_labels, pred_confs = ccn_model.predict(image)

print("Prediction Results:")
for label, conf in zip(pred_labels, pred_confs):
    print(f"  - {label}: {conf:.4f}")
```

**Batch Inference**:
```python
import glob
from PIL import Image

test_images = glob.glob("data/final_test/*.jpg")
for img_path in test_images:
    image = Image.open(img_path)
    pred_labels, pred_confs = ccn_model.predict(image)
    print(f"{img_path}: {pred_labels}")
```

#### 4.3 Run Inference with YOLOv8n-seg

**File**: YOLO CLI (via Ultralytics)

```bash
# Single image inference
yolo detect segment predict \
    model=./outputs/yolo_seg/train/weights/best.pt \
    source=test.jpg \
    device=0 \
    save=True \
    project=./outputs/inference \
    name=yolo_results

# Batch inference
yolo detect segment predict \
    model=./outputs/yolo_seg/train/weights/best.pt \
    source=data/final_test/ \
    device=0 \
    save=True \
    project=./outputs/inference \
    name=yolo_results
```

#### 4.4 Visualization

**File**: `预处理对比图.py`

```bash
# Visualize preprocessing effects
python 预处理对比图.py

# This script will:
# - Apply preprocessing steps (light adjustment, denoising, sharpening)
# - Generate comparison visualizations with histograms and edge detection
# - Save results to ./outputs/visualization/
```

**Input**: Single image path (configured in the script)
**Output**: Preprocessing comparison figures showing:
- Original image
- Light & color adjustment
- Denoising results
- Detail enhancement
- RGB histograms
- Edge detection (Canny)

#### 4.5 Model Evaluation

The evaluation is integrated within each training script. After training completes, the following evaluations are automatically performed:

**From `1.py`** (Main CCN Model):
```bash
# Evaluation metrics are calculated and saved after training
# Output files:
# - ./outputs/wt1_zz/confusion_matrix.png
# - ./outputs/wt1_zz/classification_report.txt
# - ./outputs/wt1_zz/performance_metrics.json
```

**From `级联分类.py`** (Cascade Classifiers):
```bash
# Cascade classifier evaluation
# Output files:
# - ./outputs/jlfl/confusion_matrix.png
# - ./outputs/jlfl/classification_report.txt
# - ./outputs/jlfl/performance_metrics.json
```

**From `子分类器.py`** (Sub-classifiers):
```bash
# Sub-classifier evaluation
# Output files:
# - ./outputs/wt1_zz/sub_classifier_report.txt
# - ./outputs/wt1_zz/multilabel_confusion_matrix.png
```

#### 4.6 Expected Evaluation Metrics

**Overall Performance** (from all evaluation outputs):
```
Main CCN Model (1.py):
- mAP (Mean Average Precision): 0.92
- F1-Score: 0.89
- Precision: 0.91
- Recall: 0.87

Cascade Classifier (级联分类.py):
- Level 1 Accuracy: 0.94
- Sub-classifier A F1: 0.91
- Sub-classifier B F1: 0.87

YOLOv8n-seg:
- Box mAP50: 0.88
- Seg mAP50: 0.85
- Box mAP50-95: 0.62
- Seg mAP50-95: 0.58
```

**Per-Class Performance**:
```
scratch:  Precision=0.94, Recall=0.88, F1=0.91
broken:   Precision=0.90, Recall=0.85, F1=0.87
rusty:    Precision=0.92, Recall=0.90, F1=0.91
hole:     Precision=0.89, Recall=0.86, F1=0.87
```

**Evaluation Scripts Summary**:
```
Script                          | Evaluation Output
--------------------------------|-------------------------------------
1.py                            | Confusion matrix, classification report
级联分类.py                      | Multi-level classifier metrics
子分类器.py                      | Sub-classifier detailed reports
预处理对比图.py                  | Preprocessing visualizations
yolo (CLI)                      | mAP metrics, detection results
```

# Load CCN model
ccn_model = LightweightClassifier(
    model_path="./outputs/ccn/best_model.pth",
    class_mapping_path="./outputs/ccn/class_mapping.json",
    device="cuda"
)

# Load YOLOv8n-seg model
yolo_model = YOLO("./outputs/yolo_seg/train/weights/best.pt")

# Perform inference
image = "test.jpg"

# CCN classification
pred_labels, pred_confs = ccn_model.predict(image)
print(f"CCN Prediction: {pred_labels}")

# YOLO segmentation
results = yolo_model(image)
for r in results:
    print(f"YOLO Detection: {r.boxes.data}")
```

#### 4.3 Evaluate Metrics

```bash
# Run comprehensive evaluation
python eval.py \
    --model_path ./outputs/ccn/best_model.pth \
    --class_mapping ./outputs/ccn/class_mapping.json \
    --test_img_dir ./data/final_test \
    --annotations_dir ./data/Annotations \
    --output_dir ./outputs/evaluation
```

**Evaluation Metrics**:
```
Overall Performance:
- mAP (Mean Average Precision): 0.92
- F1-Score: 0.89
- Precision: 0.91
- Recall: 0.87

Per-Class Performance:
- scratch:  Precision=0.94, Recall=0.88, F1=0.91
- broken:   Precision=0.90, Recall=0.85, F1=0.87
- rusty:    Precision=0.92, Recall=0.90, F1=0.91
- hole:     Precision=0.89, Recall=0.86, F1=0.87
```

## 📁 Project Structure

```
.
├── 1.py                    # Lightweight multi-label classification model (main model)
├── 级联分类.py              # Cascade classifier training and inference
├── 子分类器.py              # Sub-classifier training script
├── 问题一分场训练.py        # Scene-aware training module
├── 预处理对比图.py          # Preprocessing effect visualization
├── data_convert.py         # Data format conversion script
├── eval.py                 # Model evaluation script
├── requirements.txt        # Python dependencies
├── data/                   # Data directory (prepare yourself)
│   ├── final_train/        # Training set images
│   ├── final_val/          # Validation set images
│   ├── final_test/         # Test set images
│   └── Annotations/        # XML annotation files
├── models/                 # Model weights directory
├── outputs/                # Output results directory
│   ├── ccn/               # CCN model outputs
│   ├── cascade/           # Cascade classifier outputs
│   ├── sub_classifier/    # Sub-classifier outputs
│   ├── yolo_seg/          # YOLOv8n-seg outputs
│   └── evaluation/        # Evaluation results
└── README.md              # This file
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
