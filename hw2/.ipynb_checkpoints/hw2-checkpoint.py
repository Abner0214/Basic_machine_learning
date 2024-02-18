import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

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

# Fill missing values if any (if your data has missing values)
# Since you didn't mention missing values, this step may not be necessary.

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

# Build a Logistic Regression model with a higher max_iter
model = LogisticRegression(max_iter=3000)
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

# You can further explore feature relationships and perform feature engineering here if needed

# Save the shuffled train and test data to new CSV files
train_data.to_csv('data/HW2_hr-analytics_train_shuffled.csv', index=False)
test_data.to_csv('data/HW2_hr-analytics_test_shuffled.csv', index=False)

# Load the test data
test_data = pd.read_csv('data/HW2_hr-analytics_test.csv')

# Preprocess the test data (encoding, filling missing values, etc.)
test_data_encoded = pd.get_dummies(test_data, columns=['sales', 'salary'])
test_data_encoded.fillna(train_data_encoded.mean(), inplace=True)

# Use the trained model to make predictions on the test data
test_predictions = model.predict(X_test_scaled)

# Create a DataFrame to store the test predictions
test_sol = pd.DataFrame({'left': test_predictions})

# Save the test predictions to "HW2_hr-analytics_test_sol.csv"
test_sol.to_csv('data/HW2_hr-analytics_test_sol.csv', index=False)