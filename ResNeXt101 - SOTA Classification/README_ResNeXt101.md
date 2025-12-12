# LSUN Scene Classification - ResNeXt101

## 👥 Group Members
* **Vineet Sharma** - 301364822

---

## 📌 Project Overview
This project implements a Deep Learning pipeline to classify indoor scenes from the LSUN dataset using the **ResNeXt101** architecture. ResNeXt serves as a powerful feature extractor, utilizing "cardinality" (grouped convolutions) to achieve high accuracy on complex image recognition tasks.

The project classifies images into four distinct categories:
* bedroom  
* dining_room  
* kitchen  
* living_room  

### Training Strategy
We utilize a **Two-Stage Transfer Learning** approach:

1. **Stage 1 (Feature Extraction):**  
   The pre-trained ResNeXt101 backbone is frozen, and only the custom classification head is trained for 10 epochs.

2. **Stage 2 (Fine-Tuning):**  
   The last 50 layers of the backbone are unfrozen and fine-tuned for 20 epochs with a low learning rate (1e-5) to specialize the features for indoor scenes.

---

## 🛠️ Prerequisites & System Requirements

### Hardware
* GPU: NVIDIA GPU with at least **11GB VRAM** (e.g., RTX 2080 Ti).  
* RAM: 16GB+ recommended.

### Software
* Python: 3.8+  
* Framework: TensorFlow 2.10.1 (required for native Windows GPU support)

---

## 📦 Required Libraries
Install the required dependencies:

pip install tensorflow-gpu==2.10.1  
pip install numpy matplotlib seaborn pandas scikit-learn pillow  

---

## 📂 Dataset Setup

**Source:** LSUN dataset (Bedroom, Dining Room, Kitchen, Living Room)  
**Root Directory:** Create a folder named `lsun_organized_resnext` (or allow the script to create it).  

### Expected Folder Structure
Project_Root/  
│  
├── lsun_organized_resnext/  
│   ├── train/  
│   ├── val/  
│   └── test/  
│  
├── ResNeXt101_Training.ipynb  
└── README_ResNeXt101.md  

---

## 🚀 How to Run the Code

### 1. Training the Model
Open **ResNeXt101_Training.ipynb** and run the cells sequentially.

#### Configuration
* Image Size: 256 × 256  
* Batch Size: 32  
* Epochs: 10 (Stage 1) + 20 (Stage 2)

#### Execution Steps
* System Check: Verifies GPU support  
* Stage 1: Trains the classification head  
* Stage 2: Fine-tunes the unfrozen layers  

**Note:**  
Best weights are saved automatically as:

resnext101_lsun_stage2_best.h5

---

## 2. Testing / Inference

To evaluate or test new images:

1. Ensure the weights file is present:  
   resnext101_lsun_stage2_best.h5

2. Run the evaluation block or the standalone script.

Outputs include:
* Classification Report  
* Confusion Matrix  

---

## 📊 Performance Results

The ResNeXt101 model demonstrates excellent performance with strong precision across all classes.

| Metric | Value |
|--------|--------|
| Model Architecture | ResNeXt101 |
| Input Resolution | 256 × 256 pixels |
| Final Test Accuracy | **91.97%** |
| Best Val Accuracy | **~92.33%** |

### Classification Report Summary

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Bedroom      | 0.95      | 0.94   | 0.94     | 4500    |
| Dining Room  | 0.91      | 0.89   | 0.90     | 4500    |
| Kitchen      | 0.93      | 0.95   | 0.94     | 4500    |
| Living Room  | 0.90      | 0.91   | 0.90     | 4500    |
| **Accuracy** |           |        | **0.92** | **18000** |
