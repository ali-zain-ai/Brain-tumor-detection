# 🧠 Brain Tumor Detection using Deep Learning (VGG16)

This project uses **Convolutional Neural Networks (CNNs)** with **VGG16 Transfer Learning** to detect whether a brain MRI scan indicates a **tumor** or **no tumor**.  
It leverages **TensorFlow**, **Keras**, **OpenCV**, and **Scikit-learn** to preprocess MRI images, train the model, and make accurate predictions.

---

## 🚀 Project Overview

The main goal of this project is to automatically classify brain MRI images into two categories:

- 🧠 **Tumor → 1**  
- ✅ **No Tumor → 0**

A pretrained **VGG16 model** is fine-tuned on a custom brain MRI dataset to achieve **high accuracy** while minimizing **overfitting**.

---

## 📂 Dataset

You can use any dataset with the following folder structure:

```
brain_tumor_dataset/
│
├── yes/       # Images with brain tumor
│   ├── Y1.jpg
│   ├── Y2.jpg
│   └── ...
│
└── no/        # Images without brain tumor
    ├── N1.jpg
    ├── N2.jpg
    └── ...
```

Each image will be **resized to 224×224 pixels** to match the **VGG16 input requirements**.

---

## ⚙️ Model Architecture

The model uses **VGG16 (pretrained on ImageNet)** as the **feature extractor**.  
Custom layers are added on top to handle classification.

### 🧱 Architecture Summary

- Base Model: `VGG16 (include_top=False)`  
- Flatten Layer  
- Dense (128 units, ReLU activation)  
- Dropout (0.5)  
- Dense (1 unit, Sigmoid activation)

This approach uses **Transfer Learning**, allowing the model to learn efficiently even with a smaller dataset.

---

## 🧩 Dependencies

Make sure you have the following libraries installed:

```bash
pip install tensorflow keras numpy pandas opencv-python matplotlib scikit-learn
```

---

## 🖼️ Data Preprocessing

The images are:

- Loaded from their respective folders (`yes` / `no`)  
- Resized to **224x224**  
- Converted into **NumPy arrays**  
- Split into **training and testing sets** using `train_test_split()`

---

## 🧠 Training

Run the Jupyter notebook:

```bash
jupyter notebook braintumer.ipynb
```

The model will:

1. Load and preprocess all MRI images  
2. Train the model using **binary crossentropy loss**  
3. Evaluate accuracy and visualize loss/accuracy curves  

You can modify the **number of epochs** or **batch size** as needed.

---

## 🔍 Prediction Guide

After the model is trained, you can use it to make predictions on new MRI images:

```python
prediction = model.predict(new_image)
```

### 🧾 Interpretation

| Output Value | Meaning |
|:--------------|:---------|
| Closer to **0** | ✅ No Tumor Detected |
| Closer to **1** | 🧠 Tumor Detected |

---

## 📊 Visualization

The notebook includes visualizations for:

- 📈 Training vs Validation Accuracy  
- 📉 Training vs Validation Loss  
- 🧩 Random Predictions (Model Output vs Actual Labels)

---

## 🏁 Results

The trained model achieves **high accuracy** in detecting tumors from MRI scans using a relatively small dataset and pretrained weights from **VGG16**.

You can further improve accuracy by:

- Adding more diverse MRI samples  
- Using data augmentation techniques  
- Fine-tuning deeper **VGG16** layers  

---

## 📜 License

This project is released under the **MIT License** —  
you are free to use, modify, and distribute it with proper credit.

---

## 👨‍💻 Author

**Ali Zain**  
📧 Email: *mindfuelbyali@gmail.com*  
💼 AI | ML | DL | NLP Engineer  

---
