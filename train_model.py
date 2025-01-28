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

print("Preprocessing complete and data saved!")

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

print(f"Random Forest Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Save the trained model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Model trained and saved!")