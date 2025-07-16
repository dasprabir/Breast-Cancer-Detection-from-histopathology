

#  Breast Cancer Detection using Deep Learning (IDC Classification)

This repository implements a deep learning pipeline to classify histopathological image patches for detecting **Invasive Ductal Carcinoma (IDC)** — a common form of breast cancer. The project was completed as part of a hands-on Udemy course and uses **TensorFlow**, **CNN**, and **OpenCV** to build and evaluate models.

---

## 📂 Folder Structure
├── Detect_BreastCancer.ipynb # End-to-end notebook for model training & evaluation
├── train_CustomModel_32_conv_20k.ipynb # Notebook for training the custom CNN model
├── train_ResNet50_32_20k.ipynb # Notebook for training the ResNet50 model
├── utils/
│ ├── config.py # Configuration for dataset paths and settings
│ ├── conv_bc_model.py # Custom CNN model architecture
│ ├── create_dataset.py # Dataset creation and preprocessing utilities
│ └── getPaths.py # Utility to extract file paths
├── output/
│ ├── RN_weights-009-0.3958.hdf5 # Saved weights for ResNet50 model
│ ├── CM_weights-010-0.3063.hdf5 # Saved weights for Custom CNN model
│ ├── RN_TrainingHistoryPlot.png # Training history plot (ResNet50)
│ └── CM_TrainingHistoryPlot.png # Training history plot (Custom CNN)
├── sampleTest_Pictures/
│ ├── benign.png # Example test image (benign)
│ └── malignant.png # Example test image (malignant)
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
└── Kaggle Link.txt # Dataset reference (Kaggle link)


---

## 📊 Project Summary

- 📁 Dataset: Breast Histopathology Images from Kaggle  
  > https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

- 🧠 Models Implemented:
  - Custom CNN (from scratch)
  - Transfer Learning with pretrained ResNet50

- ⚙️ Techniques Used:
  - Data Augmentation
  - Image Generators
  - Model Checkpointing
  - Training and Validation Plots
  - Performance Metrics: Accuracy, Sensitivity, Specificity, AUC-ROC

---

## 🧪 Features

- Full pipeline: Load, preprocess, train, evaluate, and predict
- Custom modular code for CNN and ResNet
- Visualizations: sample images, training curves, confusion matrix
- Predict outcome from new histopathology images

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
🚀 How to Use
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/dasprabir/Breast-Cancer-Detection-from-histopathology.git
cd Breast-Cancer-Detection-from-histopathology
2. Prepare the Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

Extract and place it in an accessible path

Update the path in config.py

3. Train Models
For Custom CNN:


Run train_CustomModel_32_conv_20k.ipynb
For ResNet50:


Run train_ResNet50_32_20k.ipynb
4. Make Predictions
Use Detect_BreastCancer.ipynb to:

Load pretrained model

Upload test images (e.g., benign.png, malignant.png)

Visualize predictions

📈 Sample Results
Model	Accuracy	AUC	Sensitivity	Specificity
ResNet50	~95%	0.97	High	High
Custom CNN	~91%	0.94	High	Moderate

Note: Performance depends on number of samples and training epochs.

🧠 Skills Gained
Data preprocessing for histopathological images

CNN architecture design and tuning

Transfer learning with ResNet50

Confusion matrix and AUC-ROC analysis

Deployment using Google Colab

📄 License
This repository is for educational use only.
Dataset © Kaggle 

👤 Author
Prabir Kumar Das
Biomedical Engineering | AI in Medical Imaging
📫 dasprabirkumar530@gmail.com



