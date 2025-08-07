# ----------------------------
# Full Pipeline: Training + Saving + Real-time Prediction for T-MLP
# ----------------------------

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from google.colab import files, drive

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Step 1: Load and Prepare Data
# ----------------------------
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/My Drive/Data/bnpl_credit_data_Light.csv')

# Preprocessing
features = ['Age', 'Credit Score', 'Total_Purchase_Frequency', 'Total_Purchase_Amount', 'Age Condition', 'Rating', 'Repeat Usage']
target = 'Target'

data.fillna(0, inplace=True)
data['Age Condition'] = np.where(data['Age'] < 18, 0, 1)
data['Credit_Condition'] = np.where(data['Credit Score'] > 519, 1, 0)
data['Repeat Usage'] = data['Repeat Usage'].map({'Yes': 1, 'No': 0})

freq_cols = [f'Monthly Purchase Frequency {i}' for i in range(1, 7)]
amount_cols = [f'Monthly Purchase Amount {i}' for i in range(1, 7)]
data['Total_Purchase_Frequency'] = data[freq_cols].sum(axis=1)
data['Total_Purchase_Amount'] = data[amount_cols].sum(axis=1)

def determine_credit(row):
    if row['Credit_Condition'] == 0:
        return 0, 0
    if row['Payment Status'] == 'No':
        if row['Total_Purchase_Amount'] > 310000001:
            return 10000000, 1
        elif row['Total_Purchase_Amount'] > 150000001:
            return 5000000, 1
    else:
        if row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] > 220000000:
            return 10000000, 3
        elif row['Total_Purchase_Frequency'] > 79:
            return 10000000, 1
        elif row['Total_Purchase_Frequency'] < 80 and row['Total_Purchase_Amount'] > 110000000:
            return 5000000, 3
        elif row['Total_Purchase_Frequency'] < 80:
            return 5000000, 1
        elif row['Total_Purchase_Frequency'] < 41 and row['Total_Purchase_Amount'] < 80000001:
            return 2000000, 1
    return 0, 0

data[['Credit Amount', 'Repayment Period']] = data.apply(determine_credit, axis=1, result_type='expand')
data['Target'] = np.where(data['Credit_Condition'] & (data['Total_Purchase_Amount'] > 10), 1, 0)



# ----------------------------
# Step 2: Feature Scaling and Selection
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['Target'], test_size=0.3, random_state=42, stratify=data['Target'])

lgb_train = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1}
lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

importance = lgb_model.feature_importance()
feature_names = features
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values(by='importance', ascending=False)
top_k = 5
selected_features = importance_df['feature'].values[:top_k]

X_train_selected = pd.DataFrame(X_train, columns=feature_names)[selected_features].values
X_test_selected = pd.DataFrame(X_test, columns=feature_names)[selected_features].values

# ----------------------------
# Step 3: Define and Train T-MLP Model
# ----------------------------
class TMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, x):
        return self.net(x)

model = TMLP(input_dim=len(selected_features)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epochs = 5
batch_size = 256

train_dataset = TensorDataset(
    torch.tensor(X_train_selected, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_preds = 0

    for batch_feats, batch_y in train_loader:
        batch_feats, batch_y = batch_feats.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_feats)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_preds += (preds == batch_y).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_preds / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

# ----------------------------
# Step 4: Save Model, Scaler, and Selected Features
# ----------------------------
MODEL_PATH = '/content/tmlp_model.pth'
SCALER_PATH = '/content/scaler.pkl'
SELECTED_FEATURES_PATH = '/content/selected_features.pkl'
SCALER_FEATURES_PATH = '/content/scaler_features.pkl'

torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(selected_features, SELECTED_FEATURES_PATH)
joblib.dump(features, SCALER_FEATURES_PATH)

print("\nModel, Scaler, Selected Features, and Scaler Features saved!")

# ----------------------------
# Smart Real-Time Prediction Function (Fixed)
# ----------------------------
def smart_predict_real_time(new_data_df, batch_size=256):
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
    full_feature_list = joblib.load(SCALER_FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    for col in full_feature_list:
        if col not in new_data_df.columns:
            new_data_df[col] = 0

    new_data_df = new_data_df[full_feature_list]
    new_data_scaled = scaler.transform(new_data_df)
    new_data_selected = pd.DataFrame(new_data_scaled, columns=full_feature_list)[selected_features].values

    data_tensor = torch.tensor(new_data_selected, dtype=torch.float32).to(device)

    all_probs = []
    model.eval()
    if len(data_tensor) <= batch_size:
        with torch.no_grad():
            outputs = model(data_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    else:
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            with torch.no_grad():
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

    return np.array(all_probs)

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_predictions(y_true, y_probs, threshold=None):
    if threshold is None:
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
        ks_values = tpr - fpr
        best_ks_idx = np.argmax(ks_values)
        threshold = thresholds_roc[best_ks_idx]

    y_pred = (y_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_probs)
    conf_matrix = confusion_matrix(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    ks_statistic = np.max(np.abs(tpr - fpr))

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc_val:.4f}")
    print(f"KS Statistic: {ks_statistic:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# ----------------------------
# Example Usage 1: Predict One Record
# ----------------------------
one_record = pd.DataFrame({
    'Age': [27],
    'Credit Score': [610],
    'Total_Purchase_Frequency': [18],
    'Total_Purchase_Amount': [95000000],
    'Age Condition': [1],
    'Rating': [4.0],
    'Repeat Usage': [1]
})

pred = smart_predict_real_time(one_record)
print(f"Approval Probability for one record: {pred[0]:.4f}")
uploaded = files.upload()
incoming_data = pd.read_csv('realtime_data.csv')
incoming_data = pd.read_csv('/content/incoming_realtime_data.csv')
full_output = smart_predict_real_time(incoming_data, save_path='/content/predicted_realtime_output.csv')

