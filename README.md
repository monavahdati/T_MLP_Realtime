# 🧠 BNPL Credit Scoring with T-MLP

A complete machine learning pipeline to predict credit approval in **Buy Now Pay Later (BNPL)** systems using a custom **Tabular Multi-Layer Perceptron (T-MLP)** model.  
This project includes: preprocessing, feature engineering, model training, evaluation, model saving, and real-time prediction functionality.

---

## 🔍 Problem Statement

In BNPL systems, assessing whether a user is eligible for a credit offer is critical. We predict credit approval based on behavioral and demographic features using a deep learning model optimized for tabular data.

---

## 🚀 Project Highlights

✅ LightGBM-based feature selection  
✅ Deep T-MLP neural network (BatchNorm + Dropout)  
✅ Real-time smart inference function  
✅ Evaluation with AUC, KS, and PRC  
✅ SHAP-ready architecture  
✅ Clean modular structure, ready for deployment  

---

## 🧱 Model Architecture

The `TMLP` model uses a deep neural network with normalization and dropout to handle tabular data effectively:

Input Features (Top 5 from LGBM)
↓
Linear(128) → ReLU → BatchNorm → Dropout(0.3)
↓
Linear(64) → ReLU → BatchNorm → Dropout(0.2)
↓
Linear(2) → Softmax

yaml
Copy
Edit

---

## 🧪 Dataset Features

Main features used for prediction:

- `Age`
- `Credit Score`
- `Total_Purchase_Frequency` *(sum of 6 months)*
- `Total_Purchase_Amount` *(sum of 6 months)*
- `Age Condition` *(binary: under/over 18)*
- `Rating`
- `Repeat Usage` *(binary)*

Target variable: `Target` (binary classification: 1 = approve, 0 = reject)

---

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score
- ROC-AUC
- KS Statistic
- Confusion Matrix

Best KS threshold is auto-selected for binary decision-making.

---

## 📈 Visualizations

The project includes:

- ROC and Precision-Recall curves
- KS Curve
- Confusion matrix heatmap
- Train loss and accuracy plots
- Gain & Lift charts

---

## 🧠 SHAP Ready

The structure allows easy integration with SHAP for feature attribution and interpretability.

---

## 🧪 Real-Time Inference

Includes a utility function:

```python
def smart_predict_real_time(new_data_df, batch_size=256):
    ...
    return np.array(probabilities)