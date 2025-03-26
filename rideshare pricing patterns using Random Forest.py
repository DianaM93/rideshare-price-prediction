# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 02:00:49 2025

@author: Giacomo
"""

#%% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#%% Load the dataset
file_path = 'rideshare_kaggle.csv'  # Make sure this file is in your working directory
df = pd.read_csv(file_path)

#%% Data Cleaning & Filtering
df = df.dropna(subset=['price'])
df = df[df['price'] < 100]
df = df[df['name'] == 'UberX']

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.day_name()

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['time_of_day'] = df['hour_of_day'].apply(get_time_of_day)

#%% Plot 1: Ride Demand by Hour
plt.figure(figsize=(10,5))
sns.countplot(x='hour_of_day', data=df, palette='Blues')
plt.title('Ride Demand by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rides')
plt.xticks(range(0, 24))
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#%% Plot 2: Average Surge by Hour
plt.figure(figsize=(10,5))
avg_surge = df.groupby('hour_of_day')['surge_multiplier'].mean().reset_index()
sns.lineplot(data=avg_surge, x='hour_of_day', y='surge_multiplier', marker='o')
plt.title('Average Surge Multiplier by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Average Surge')
plt.grid(True)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

#%% Plot 3: Price by Weather
plt.figure(figsize=(12,6))
sns.boxplot(x='short_summary', y='price', data=df)
plt.title('Price Distribution by Weather Condition')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%% Plot 4: Surge Multiplier > 1 by Weather
plt.figure(figsize=(12,6))
df_surge = df[df['surge_multiplier'] > 1.0]
if not df_surge.empty:
    sns.boxplot(x='short_summary', y='surge_multiplier', data=df_surge)
    plt.title('Surge Multiplier > 1 by Weather Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No surge data above 1.0 for plotting.")

#%% Encode data for model
df_encoded = pd.get_dummies(df[['distance', 'hour_of_day', 'temperature', 'short_summary']], drop_first=True)
target = df['price']

X_train, X_test, y_train, y_test = train_test_split(df_encoded, target, test_size=0.2, random_state=42)

#%% Train & evaluate model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("✅ Random Forest Regressor Results:")
print("MAE:", round(rf_mae, 2))
print("R² Score:", round(rf_r2, 4))
