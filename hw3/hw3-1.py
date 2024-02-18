import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import shap

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read the original training data
original_train_data = pd.read_csv('data/HW2_hr-analytics_train.csv')

# Shuffle the data every time when you run the code
shuffled_train_data = original_train_data.sample(frac=1)  # Shuffle without specifying a random state

# Split the shuffled data into train (80%) and test (20%)
train_data, test_data = train_test_split(shuffled_train_data, test_size=0.2)

# Encode non-numeric columns
train_data_encoded = pd.get_dummies(train_data, columns=['sales', 'salary'])
test_data_encoded = pd.get_dummies(test_data, columns=['sales', 'salary'])

# Check for missing values in the training data
missing_train = train_data_encoded.isna().any()
if missing_train.any():
    print("There are missing values in the training data.")
    print(missing_train)
    train_data_encoded.fillna(train_data_encoded.median(), inplace=True)
    print("Train data after filling missing values with median:")
    print(train_data_encoded)
else:
    print("No missing values in the training data.")

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data_encoded[['satisfaction_level', 'last_evaluation', 'number_project',
                                                        'average_montly_hours', 'time_spend_company', 'Work_accident',
                                                        'promotion_last_5years', 'sales_IT', 'sales_RandD',
                                                        'sales_accounting', 'sales_hr', 'sales_management',
                                                        'sales_marketing', 'sales_product_mng', 'sales_sales',
                                                        'sales_support', 'sales_technical', 'salary_high',
                                                        'salary_low', 'salary_medium']])
X_test_scaled = scaler.transform(test_data_encoded[['satisfaction_level', 'last_evaluation', 'number_project',
                                                    'average_montly_hours', 'time_spend_company', 'Work_accident',
                                                    'promotion_last_5years', 'sales_IT', 'sales_RandD',
                                                    'sales_accounting', 'sales_hr', 'sales_management',
                                                    'sales_marketing', 'sales_product_mng', 'sales_sales',
                                                    'sales_support', 'sales_technical', 'salary_high',
                                                    'salary_low', 'salary_medium']])

# Build a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train_scaled, train_data_encoded['left'])

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(test_data_encoded['left'], y_pred)
accuracy = accuracy_score(test_data_encoded['left'], y_pred)

# Format the accuracy with five digits after the decimal point
formatted_accuracy = '{:.5f}'.format(accuracy)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", formatted_accuracy)

# Calculate feature importances
feature_importances = model.feature_importances_

# Print the feature importances
print("Feature Importances:")
for feature, importance in zip(train_data_encoded.columns[1:], feature_importances):
    print(f"{feature}: {importance:.5f}")

# Use SHAP for feature importance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# Summarize the feature importance
shap.summary_plot(shap_values, X_test_scaled, feature_names=train_data_encoded.columns)
