
# LSUN Scene Classification - ConvNeXt-Tiny

## 👥 Group Members
* **Namneet Kaur**

## 📌 Project Overview
This project deploys a **ConvNeXt-Tiny** model, a modern "pure ConvNet" architecture inspired by Vision Transformers (ViT). We utilize the `keras_cv_attention_models` library to access this specific architecture for high-performance scene classification on the LSUN dataset.

The project classifies images into four distinct categories:
* `bedroom`
* `dining_room`
* `kitchen`
* `living_room`

**Key Strategy:** A robust **Two-Stage Training** approach is used:
* **Stage 1 (Head Training):** The backbone is frozen, and only the classification head is trained for 10 epochs.
* **Stage 2 (Fine-Tuning):** All layers are unfrozen and fine-tuned with a low learning rate (`1e-5`) to maximize accuracy.

## 🛠️ Prerequisites & System Requirements

### Hardware
* **GPU:** NVIDIA GPU with at least **11GB VRAM** (e.g., RTX 2080 Ti).
* **RAM:** 16GB+ System RAM.

### Software
* **Python:** 3.8+
* **Framework:** TensorFlow 2.10.1 (Required for native Windows GPU support).

## 📦 Required Libraries
**Important:** This model requires a specific external library (`keras_cv_attention_models`) in addition to standard TensorFlow.

Install the dependencies via pip:

```bash
# Core Deep Learning
pip install tensorflow-gpu==2.10.1

# Specific Model Library
pip install keras_cv_attention_models

# Data Processing & Metrics
pip install numpy matplotlib seaborn pandas scikit-learn pillow
```

## 📂 Dataset Setup
Source: LSUN dataset samples (Bedroom, Dining Room, Kitchen, Living Room).
Root Directory: Create a folder named `lsun_organized_resnext` (reusing the organized folder from previous experiments saves disk space).

Structure: The code expects the following directory tree:

```
Project_Root/
│
├── lsun_organized_resnext/   <-- Main Data Folder
│   ├── train/                <-- Training Images (grouped by class)
│   ├── val/                  <-- Validation Images
│   └── test/                 <-- Test Images
│
├── ConvNeXt_Training.ipynb
└── README_ConvNeXt.md
```

## 🚀 How to Run the Code

### 1. Training the Model
Open the training notebook (`ConvNeXt_Training.ipynb`) and execute the cells sequentially.

Configuration:
- Image Size: 256 x 256
- Batch Size: 32
- Epochs: 10 (Stage 1) + 30 (Stage 2)

Execution Steps:
- System Check: Verifies GPU availability and keras_cv_attention_models installation.
- Stage 1: Trains the head (frozen backbone).
- Stage 2: Fine-tunes the entire model.

Note: The model saves the best weights automatically to `convnext_tiny_lsun_stage2_best.h5` whenever validation accuracy improves.

### 2. Testing / Inference
To evaluate the model or run predictions on new images:
- Ensure the weights file `convnext_tiny_lsun_stage2_best.h5` is present.
- Run the evaluation block in the notebook or the standalone script.

The script will generate a classification report and a confusion matrix.

## 📊 Performance Results
The ConvNeXt-Tiny model achieves state-of-the-art performance, surpassing standard ResNet architectures on this dataset.

### Metrics
| Metric | Value |
|--------|-------|
| Model Architecture | ConvNeXt-Tiny |
| Input Resolution | 256 x 256 pixels |
| Final Test Accuracy | 94.14% |
| Best Val Accuracy | ~94.55% |

### Classification Report Summary
The model exhibits exceptional precision, particularly for the Bedroom class (97% precision).

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Bedroom      | 0.97      | 0.96   | 0.97     | 4500    |
| Dining Room  | 0.92      | 0.91   | 0.92     | 4500    |
| Kitchen      | 0.94      | 0.95   | 0.95     | 4500    |
| Living Room  | 0.93      | 0.94   | 0.94     | 4500    |
| **Accuracy** | **0.94**  | —      | —        | **18000** |
```
