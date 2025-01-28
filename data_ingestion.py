import pandas as pd

# Provide the full path to the file you uploaded
file_path = "/Users/yugbhavsar/Downloads/AI_Sales_Forecasting/data/OnlineRetail.csv"

# Load your dataset
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Display basic information about the dataset
print("\nDataset Info:")
data.info()

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())
