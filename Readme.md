

# **DeformPTM: Pre-trained constitutive foundation model**


This repository provides a full pipeline for  **Leave-One-Out (LOO) evaluation**, **stress–strain curve prediction**, and **inverse thermomechanical processing (TMP) optimization** across alloys.

The framework integrates:

* **DeformPTM fine-tuning using experimental data**
* **High-accuracy stress–strain prediction**
* **PSO-based inverse TMP search**
* **Baseline comparison using ANN and SVM**

This repository supports research in **alloy deformation modeling**, **mechanistic interpretability**, and **machine learning–guided TMP design**.

---

## **Key Features**

### **1. DeformPTM Fine-Tuning Pipeline**

* Loads pretrained DeformPTM model
* Freezes Bi-LSTM encoder, fine-tunes latent projections + decoder
* Fine-tunes DeformPTM on experimental data
* Outputs the predicted stress–strain curves
---

### **3. Leave-One-Out Cross Validation (LOO)**

For each deformation condition:

* Fit constitutive model on training curves
* Fine-tune DeformPTM on training data
* Predict the held-out condition
* Compute evaluation metrics:

  * **MSE**
  * **R²**
  * **MAPE**
* Export predicted vs. true CSV curves

---

### **4. Stress–Strain Inference**

Given:

* Temperature T
* Strain rate $\dot{ε}$

The pipeline can:

1. Feed it into fine-tuned DeformPTM
2. Produce a final mechanistically guided stress-strain curve

---

### **5. PSO-Based Inverse TMP Optimization**

Given a **target stress–strain curve**:

* Use PSO to search the 2-D space *(T, strain rate)*
* Objective: minimize MSE w.r.t. target curve
* Output:

  * Best processing parameters
  * Convergence curve
  * Optimized deformation path

This enables:
✔️ Generating TMP routes that realize a desired deformation response
✔️ Demonstrating DeformPTM’s inverse-design capability

---

### **6. Baseline Models: ANN & SVM**

Included for fair comparison with traditional ML models.

* ANN with two hidden layers
* SVM with RBF kernel
* LOO evaluation identical to DeformPTM
* Saves:

  * Prediction CSV
  * Per-condition plots
  * Bar charts of MSE and R²

---

## **Directory Structure**

```
├── alloy_dataset/
│   ├── AQ80/                # Raw stress–strain CSVs
│   ├── AQ80_LOO/            # Saved LOO-calculated curves
│   └── ...
│
├── cal_data/                # Polynomial fitted curves
├── pred_data/               # DeformPTM predicted curves
├── pred_csv/                # ANN predicted curves
├── pred_csv_svm/            # SVM predicted curves
│
├── DeformPTM.pth            # Pretrained model
├── DeformPTM_fintuned.pth   # Fine-tuned model
│
├── *.ipynb                     # Core code (single notebook file)
└── README.md                # (this file)
```



## **Contact**

For questions or collaboration:
**Jiaxuan Ma**
Email: jxma@sjtu.edu.cn
