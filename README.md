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
🟢 Automatically:

Pads missing columns

Applies stored scaler

Selects LGBM top-K features

Returns credit approval probability

📁 Project Structure
bash
Copy
Edit
BNPL-TMLP/
│
├── model.py                  # TMLP model definition
├── train_pipeline.py         # Full pipeline: preprocessing + training + saving
├── predict.py                # Smart inference script
├── requirements.txt
├── README.md
└── saved/
    ├── tmlp_model.pth
    ├── scaler.pkl
    ├── selected_features.pkl
    └── scaler_features.pkl
💾 Output Artifacts
After training, the following files will be saved:

File	Description
tmlp_model.pth	Trained PyTorch model
scaler.pkl	StandardScaler used for feature normalization
selected_features.pkl	Top K features chosen by LightGBM
scaler_features.pkl	Full feature list used in scaling

🧪 Example Inference
python
Copy
Edit
# Predict one user
df = pd.DataFrame({
    'Age': [28],
    'Credit Score': [590],
    'Total_Purchase_Frequency': [35],
    'Total_Purchase_Amount': [100000000],
    'Age Condition': [1],
    'Rating': [3.5],
    'Repeat Usage': [1]
})

pred_prob = smart_predict_real_time(df)
print(f"Approval Probability: {pred_prob[0]:.4f}")
📦 Requirements
bash
Copy
Edit
torch
pandas
numpy
lightgbm
scikit-learn
matplotlib
seaborn
joblib
Install via:

bash
Copy
Edit
pip install -r requirements.txt
📌 Future Improvements
Add SHAP explanations per prediction

Convert to REST API using FastAPI or Flask

Integrate credit risk thresholds into a dashboard

🪪 License
MIT License

🤝 Acknowledgments
This project is part of a broader effort to improve credit decision-making in digital financial systems using deep learning and explainability.