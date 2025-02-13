# Sales Forecasting Project

This project focuses on predicting sales using machine learning techniques. It involves data ingestion, preprocessing, model training, and evaluation using the `RandomForestRegressor` from the `scikit-learn` library.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
   

---

## Project Overview

The goal of this project is to predict sales (specifically, `UnitPrice`) using historical sales data. The dataset used is `OnlineRetail.csv`, which contains transactional data for an online retail store. The project involves the following steps:
1. **Data Ingestion**: Load and inspect the dataset.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numeric features.
3. **Model Training**: Train a `RandomForestRegressor` model to predict sales.
4. **Model Evaluation**: Evaluate the model using Mean Squared Error (MSE) and R-squared (R²).
5. **Feature Importance**: Visualize the importance of features in the model.

## Project Structure

Sales_Forecasting

data
- OnlineRetail.csv
           
scripts
 - data_ingestion.py         
 - preprocessing.py          
 - train_model.py
              
outputs 
  - preprocessed_data.pkl
  - random_forest_model.pkl   
 
visulization
  - plot.py
