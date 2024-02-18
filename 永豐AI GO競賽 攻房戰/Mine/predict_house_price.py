import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pyproj
from geopy.distance import geodesic
import math
import warnings
from joblib import Parallel, delayed

# Ignore FutureWarnings related to is_sparse and pd.SparseDtype
warnings.filterwarnings('ignore', category=FutureWarning)

# Load your house_data (ensure it's properly translated and preprocessed)
house_data = pd.read_csv('training_data.csv')

# Save the 'ID' column for later use
ids = house_data['ID']

# Encode categorical variables (one-hot encoding)
house_data = pd.get_dummies(house_data, columns=['縣市', '鄉鎮市區', '使用分區', '主要用途', '主要建材', '建物型態'])

# Drop non-numeric columns
house_data.drop(columns=['ID', '路名', '備註'], inplace=True)

# Define target and features
X = house_data.drop(columns=['單價'])  # Replace '單價' with the English translation of your target column
y = house_data['單價']  # Replace '單價' with the Englishouse_datah translation of your target column


# Convert TWD97 coordinates to decimal longitude and latitude
twd97 = pyproj.Proj(init='epsg:3826')  # TWD97 projection
wgs84 = pyproj.Proj(init='epsg:4326')  # WGS84 projection
house_data['longitude'], house_data['latitude'] = pyproj.transform(twd97, wgs84, house_data['橫坐標'].values, house_data['縱坐標'].values)

# # External data folder path
# external_data_folder = 'external_data'






# import subprocess

# def calculate_distance_c(coord1, coord2):
#     lat1, lon1 = coord1
#     lat2, lon2 = coord2

#     # Run the compiled C code as a subprocess
#     result = subprocess.run(['./distance_calculator', str(lat1), str(lon1), str(lat2), str(lon2)],
#                              stdout=subprocess.PIPE, text=True)

#     try:
#         # Try to convert the output to a float
#         distance = float(result.stdout.strip())
#         return distance
#     except ValueError:
#         # Handle the case where conversion to float fails
#         print("Error: Could not convert output to float. Returning default distance.")
#         return 0.0  # You can replace this with a default value of your choice









# # Iterate through each row in the house data
# for index, house_row in house_data.iterrows():
#     house_coord = (house_row['latitude'], house_row['longitude'])

#     # Initialize counts for each category
#     atm_count = 0
#     convenience_store_count = 0
#     bus_stop_count = 0
#     middle_school_count = 0
#     elementary_school_count = 0
#     university_count = 0
#     metro_station_count = 0
#     train_station_count = 0
#     bike_station_count = 0
#     post_office_count = 0
#     medical_institution_count = 0
#     financial_institution_count = 0
#     high_school_count = 0

#     # Iterate through each row in external dataframes and calculate distance
#     for file_name in os.listdir(external_data_folder):
#         file_path = os.path.join(external_data_folder, file_name)
#         external_row = pd.read_csv(file_path)
        
#         for ext_index, ext_row in external_row.iterrows():
#             ext_coord = (ext_row['lat'], ext_row['lng'])
#             # Check if both house and external coordinates are valid
#             if not any(math.isnan(coord) for coord in house_coord) and not any(math.isnan(coord) for coord in ext_coord):
#                 distance = calculate_distance_c(house_coord, ext_coord)

                
#             # Count if the distance is less than 800 meters
#             if distance < 800:

#                 print(distance)####################

#                 if 'ATM' in file_name:
#                     atm_count += 1
#                 elif '便利商店' in file_name:
#                     convenience_store_count += 1
#                 elif '公車站點' in file_name:
#                     bus_stop_count += 1
#                 elif '國中' in file_name:
#                     middle_school_count += 1
#                 elif '國小' in file_name:
#                     elementary_school_count += 1
#                 elif '大學' in file_name:
#                     university_count += 1
#                 elif '捷運站點' in file_name:
#                     metro_station_count += 1
#                 elif '火車站點' in file_name:
#                     train_station_count += 1
#                 elif '腳踏車站點' in file_name:
#                     bike_station_count += 1
#                 elif '郵局據點' in file_name:
#                     post_office_count += 1
#                 elif '醫療機構' in file_name:
#                     medical_institution_count += 1
#                 elif '金融機構' in file_name:
#                     financial_institution_count += 1
#                 elif '高中' in file_name:
#                     high_school_count += 1

#     # Add new features to house data
#     house_data.at[index, 'atm_count'] = atm_count
#     house_data.at[index, 'convenience_store_count'] = convenience_store_count
#     house_data.at[index, 'bus_stop_count'] = bus_stop_count
#     house_data.at[index, 'middle_school_count'] = middle_school_count
#     house_data.at[index, 'elementary_school_count'] = elementary_school_count
#     house_data.at[index, 'university_count'] = university_count
#     house_data.at[index, 'metro_station_count'] = metro_station_count
#     house_data.at[index, 'train_station_count'] = train_station_count
#     house_data.at[index, 'bike_station_count'] = bike_station_count
#     house_data.at[index, 'post_office_count'] = post_office_count
#     house_data.at[index, 'medical_institution_count'] = medical_institution_count
#     house_data.at[index, 'financial_institution_count'] = financial_institution_count
#     house_data.at[index, 'high_school_count'] = high_school_count


# Save the updated house data
house_data.to_csv('updated_house_data.csv', index=False)


# Number of iterations and test size
num_iterations = 15  # You can change this to the number of iterations you want
test_size = 0.2

avg_mape = 0

predicted_prices = pd.DataFrame(columns=['ID', 'predicted_price'])

for i in range(num_iterations):
    # Split the data into training and testing sets with a different random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    
    # Create an XGBoost model and fit it to the training data
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate MAPE using NumPy for a more concise code
    mape = np.mean(np.abs((y_test - y_pred) / y_test) * 100)
    avg_mape += mape

    print(f"MAPE {i+1}: {mape}")

    # Create a DataFrame with 'ID' and 'predicted_price'
    iteration_ids = ids[y_test.index]
    iteration_predicted_prices = pd.DataFrame({'ID': iteration_ids, 'predicted_price': y_pred})
    predicted_prices = pd.concat([predicted_prices, iteration_predicted_prices])

avg_mape /= num_iterations
print("/")
print(f"{num_iterations} times average MAPE: {avg_mape}")

# Save the predicted prices to a CSV file
predicted_prices.to_csv('predicted_prices.csv', index=False)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create an XGBoost model and fit it to all the data
model = XGBRegressor()
model.fit(X, y)

# Make predictions on all the data
y_pred = model.predict(X)

# Calculate MAPE using NumPy for a more concise code
mape = np.mean(np.abs((y - y_pred) / y) * 100)
print(f"MAPE: {mape}")

# Create a DataFrame with 'ID' and 'predicted_price'
predicted_prices = pd.DataFrame({'ID': ids, 'predicted_price': y_pred})

#Save the predicted prices to a CSV file
predicted_prices.to_csv('predicted_prices.csv', index=False)



# Load your test data 
test_public_dataset = pd.read_csv('public_dataset.csv')

# Create a DataFrame with 'ID' and 'predicted_price'
test_public_dataset_ids = test_public_dataset['ID']  # Assuming 'ID' is the column name for IDs in the test data

# Preprocess the test data
test_public_dataset = pd.get_dummies(test_public_dataset, columns=['縣市', '鄉鎮市區', '使用分區', '主要用途', '主要建材', '建物型態'])
test_public_dataset.drop(columns=['ID', '路名', '備註'], inplace=True)


# Ensure the test data has the same columns as the training data
missing_columns = set(X.columns) - set(test_public_dataset.columns)
for column in missing_columns:
    test_public_dataset[column] = 0  # Set missing columns to zero or some default value

# Reorder columns to match the order in the training data
test_public_dataset = test_public_dataset[X.columns]


# Make predictions on the test data
test_predictions = model.predict(test_public_dataset)

predicted_test_public_dataset_prices = pd.DataFrame({'ID': test_public_dataset_ids, 'predicted_price': test_predictions})

# Save the predicted prices for the test data to a CSV file
predicted_test_public_dataset_prices.to_csv('predicted_public_dataset_prices.csv', index=False)






# Load your test data 
test_private_dataset = pd.read_csv('private_dataset.csv')

# Create a DataFrame with 'ID' and 'predicted_price'
test_private_dataset_ids = test_private_dataset['ID']  # Assuming 'ID' is the column name for IDs in the test data

# Preprocess the test data
test_private_dataset = pd.get_dummies(test_private_dataset, columns=['縣市', '鄉鎮市區', '使用分區', '主要用途', '主要建材', '建物型態'])
test_private_dataset.drop(columns=['ID', '路名', '備註'], inplace=True)


# Ensure the test data has the same columns as the training data
missing_columns = set(X.columns) - set(test_private_dataset.columns)
for column in missing_columns:
    test_private_dataset[column] = 0  # Set missing columns to zero or some default value

# Reorder columns to match the order in the training data
test_private_dataset = test_private_dataset[X.columns]



# Make predictions on the test data
test_predictions = model.predict(test_private_dataset)

predicted_test_private_dataset_prices = pd.DataFrame({'ID': test_private_dataset_ids, 'predicted_price': test_predictions})

# Save the predicted prices for the test data to a CSV file
predicted_test_private_dataset_prices.to_csv('predicted_private_dataset_prices.csv', index=False)