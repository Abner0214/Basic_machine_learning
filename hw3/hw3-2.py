import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load and split the data
data = pd.read_csv('data/HW3_creditcard.csv')
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Decision Tree Model
# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Decision Tree Model:")
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-Score:", f1)
print("AUROC:", roc_auc)

# 3. Check class balance
class_0_count = np.sum(y == 0)
class_1_count = np.sum(y == 1)

print("/")
print("Check class balance:")
print("Class 0 count:", class_0_count)
print("Class 1 count:", class_1_count)

# 4. Improve recall
# 4.1. Change class weights
class_weights = {   0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 1,
                    13: 1,
                    14: 1,
                    15: 1,
                    16: 1,
                    17: 1,
                    18: 1,
                    19: 1,
                    20: 1,
                    21: 1,
                    22: 1,
                    23: 1,
                    24: 1,
                    25: 1,
                    26: 1,
                    27: 1,
                    28: 1,
                    29: 1,
                    30: 1,
                    31: 1
                }

dt_model_weighted = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model_weighted.fit(X_train, y_train)
y_pred_weighted = dt_model_weighted.predict(X_test)

# Calculate evaluation metrics for the weighted model
conf_matrix_weighted = confusion_matrix(y_test, y_pred_weighted)
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
recall_weighted = recall_score(y_test, y_pred_weighted)
precision_weighted = precision_score(y_test, y_pred_weighted)
f1_weighted = f1_score(y_test, y_pred_weighted)
roc_auc_weighted = roc_auc_score(y_test, y_pred_weighted)

print("/")
print("Decision Tree Model with Weighted Classes:")
print("Confusion Matrix:")
print(conf_matrix_weighted)
print("Accuracy:", accuracy_weighted)
print("Recall:", recall_weighted)
print("Precision:", precision_weighted)
print("F1-Score:", f1_weighted)
print("AUROC:", roc_auc_weighted)


# 4.2. Decision Tree Model with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

dt_model_smote = DecisionTreeClassifier(random_state=42)
dt_model_smote.fit(X_resampled, y_resampled)
y_pred_smote = dt_model_smote.predict(X_test)

# Calculate evaluation metrics for the SMOTE model
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
accuracy_smote = accuracy_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)
roc_auc_smote = roc_auc_score(y_test, y_pred_smote)

print("/")
print("Decision Tree Model with SMOTE:")
print("Confusion Matrix:")
print(conf_matrix_smote)
print("Accuracy:", accuracy_smote)
print("Recall:", recall_smote)
print("Precision:", precision_smote)
print("F1-Score:", f1_smote)
print("AUROC:", roc_auc_smote)

# 5. Use XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)

print("/")
print("XGBoost Model:")
print("Confusion Matrix:")
print(conf_matrix_xgb)
print("Accuracy:", accuracy_xgb)
print("Recall:", recall_xgb)
print("Precision:", precision_xgb)
print("F1-Score:", f1_xgb)
print("AUROC:", roc_auc_xgb)


warnings.filterwarnings("default", category=FutureWarning)
