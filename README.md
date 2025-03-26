
# 🚕 Rideshare Pricing Prediction using Data Mining

This repository contains the final code and visualizations for a business analytics project completed for the **MSc Business Analytics - Data Mining for Business** module.

## 📊 Project Overview

The aim of this project was to analyze factors influencing rideshare prices and apply predictive data mining techniques to forecast ride fares. After exploratory analysis and model comparison, a **Random Forest Regressor** was selected as the final model due to its strong performance on cleaned data.

## 📁 Files in This Repository

- `rideshare_model.py`: Python code with preprocessing, visualizations, and the final model (run in Spyder)
- `Figure_1.1-Ride Demand by Hour.png`: Demand by hour of day (filtered dataset)
- `Figure_2.2-Average Surge by Hour.png`: Average surge multiplier by hour
- `Figure_3.3-Price by Weather.png`: Price distribution by weather condition

## 📂 Dataset Summary

- Source: Kaggle Rideshare Dataset
- Size: ~690,000 rows, 57 columns
- Key variables: distance, weather, hour of day, surge multiplier, price
- Focused subset: UberX rides under $100

## 🧪 Methodology

1. **Data Preprocessing**
   - Filtered UberX rides
   - Removed outliers
   - Encoded categorical features

2. **Exploratory Data Analysis**
   - Identified patterns in pricing and demand
   - Assessed surge pricing behavior

3. **Model Training**
   - Multiple models were tested
   - Final model: **Random Forest Regressor**
   - Performance:  
     - MAE: ~$1.14  
     - R² Score: ~0.46

## 📈 Business Value

- Helps rideshare platforms understand **key price drivers**
- Supports **dynamic pricing** strategies
- Provides insights into **demand fluctuations** by hour and weather

## 📎 Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn

## 👤 Author

DIANA-MARIANA SUHAN  
MSc Business Analytics – Data Mining for Business  
University of Greenwich


---

