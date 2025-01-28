import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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