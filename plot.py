import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Provide the full path to the file you uploaded
file_path = "/Users/yugbhavsar/Downloads/AI_Sales_Forecasting/data/OnlineRetail.csv"

# Load your dataset
data = pd.read_csv(file_path)

# Handle missing values (you can customize this based on your dataset)
data.fillna(0, inplace=True)

# Drop unnecessary columns (e.g., InvoiceNo, Description)
# Adjust this based on your dataset
data = data.drop(columns=['InvoiceNo', 'Description', 'InvoiceDate'])

# Encode categorical columns (e.g., StockCode, Country)
label_encoders = {}
for column in ['StockCode', 'Country']:
    if column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Define feature columns (X) and target column (y)
X = data.drop(columns=['UnitPrice'])  
y = data['UnitPrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (only numeric columns)
scaler = StandardScaler()
numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Save the preprocessed data and scaler
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoders': label_encoders
    }, f)

# Debugging: Check shapes of the data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Initialize the Random Forest model with fewer trees and parallel processing
rf_model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1)

# Train the model on the training data
print("Training the model...")
rf_model.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions...")
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the trained model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

import matplotlib.pyplot as plt

# Get feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Sort feature importances and feature names in descending order
sorted_idx = feature_importances.argsort()[::-1]  # Sort indices in descending order
sorted_feature_importances = feature_importances[sorted_idx]
sorted_feature_names = feature_names[sorted_idx]

# Plot feature importances
plt.figure(figsize=(10, 6))
top_n = 5  # Highlight the top 5 features
colors = ['skyblue' if i < top_n else 'gray' for i in range(len(sorted_feature_names))]
plt.barh(sorted_feature_names, sorted_feature_importances, color=colors)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importances - Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.6)

# Add annotations
for i, v in enumerate(sorted_feature_importances):
    plt.text(v, i, f"{v:.4f}", color='black', va='center')

plt.show()