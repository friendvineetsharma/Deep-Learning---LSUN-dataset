# LSUN Scene Classification - EfficientNetV2M

## 👥 Group Members
- **Vineet Sharma - 301364822**

---

## 📌 Project Overview
This project implements a state-of-the-art **Transfer Learning** pipeline to classify complex indoor scenes from the **LSUN dataset** using the **EfficientNetV2M** architecture, known for its superior parameter efficiency and accuracy.

The project distinguishes between four specific room categories:

- bedroom  
- dining_room  
- kitchen  
- living_room  

### Key Strategy
A robust **Two-Stage Training** approach is used:

1. **Feature Extraction**  
   Train only the custom classification head while the backbone is frozen.

2. **Fine-Tuning**  
   Unfreeze the top layers of the backbone and train with a very low learning rate to adapt features specifically to room interiors.

---

## 🛠️ Prerequisites & System Requirements

### Hardware
- GPU: NVIDIA GPU with at least **11GB VRAM** (e.g., RTX 2080 Ti, RTX 3080/4080 or better).  
  _Note: EfficientNetV2M at 384×384 resolution is memory intensive._
- RAM: 16GB+ recommended.

### Software
- Python: 3.8, 3.9, or 3.10  
- OS: Windows 10/11 or Linux  

---

## 📦 Required Libraries
Install the exact dependencies required to run the notebook and scripts:

pip install tensorflow==2.10.1  
pip install numpy matplotlib seaborn pandas pillow  
pip install scikit-learn  

---

## 📂 Dataset Setup

Source: Retrieve the LSUN dataset images for the four categories listed above.  
Root Directory: Create a folder named **Dataset** in the same directory as the notebook.

### Folder Structure
Project_Root/  
│  
├── Dataset/  
│   ├── bedroom/        (~30,000 images)  
│   ├── dining_room/    (~30,000 images)  
│   ├── kitchen/        (~30,000 images)  
│   └── living_room/    (~30,000 images)  
│  
├── EfficientNetV2M_Training.ipynb  
└── README.md  

---

## 🚀 How to Run the Code

### 1. Training the Model
Open **EfficientNetV2M_Training.ipynb** in Jupyter Notebook or VS Code.

#### Step 1: Setup & Validation
- Verify TensorFlow detects your GPU.
- Validate dataset folder and image counts.

#### Step 2: Data Organization
Creates **lsun_organized_v2m**, splitting data into:
- Train: 70%  
- Validation: 15%  
- Test: 15%  

#### Step 3: Stage 1 Training (Feature Extraction)
- Train only the classification head  
- 15 epochs  
- Batch Size = 16  
- Backbone frozen  

#### Step 4: Stage 2 Training (Fine-Tuning)
- Unfreeze model  
- Train 50 epochs  
- Batch Size = 8  
- Learning Rate = 1e-6  

**Output Weights:**  
EfficientNetV2M_Stage2_FineTuned_best.h5

---

## 2. Testing / Inference

1. Ensure weights file is in the project root:
   EfficientNetV2M_Stage2_FineTuned_best.h5

2. Place a test image in the project root:
   Kitchen_1.jpg

3. Run the inference script:
   Test_EfficientNet.py

---

## 📊 Performance Results

Final fine-tuned model performance on unseen **Test Set (18,000 images):**

| Metric | Value |
|-------|--------|
| Model Architecture | EfficientNetV2M |
| Input Resolution | 384 × 384 |
| Test Accuracy | ~94.12% |
| Loss | 0.2197 |

### Classification Report Summary
The model shows strong precision and recall across all classes, with **Bedroom** achieving the highest performance.
