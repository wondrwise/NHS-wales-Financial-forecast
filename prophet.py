# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:50:04 2025

@author: edwar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL


def preprocess_nhs_expenditure_percentage(df):
    """
    Prepares NHS expenditure dataset for analysis.
    Steps:
    - Removes unnecessary columns.
    - Filters data to ensure only relevant categories remain.
    - Checks for missing values and inconsistencies.
    """
    # Drop unnecessary columns (if any exist similar to other datasets)
    lhb_columns = [ 'Category breakdown',
        "2020-21 LHB primary", "2020-21 LHB secondary", "2020-21 LHB and PHW other",
        "2021-22 LHB primary", "2021-22 LHB secondary", "2021-22 LHB and PHW other",
        "2022-23 LHB primary", "2022-23 LHB secondary", "2022-23 LHB and PHW other"
    ]
    
    df_cleaned = df.drop(columns=[col for col in lhb_columns if col in df.columns], errors='ignore')

    # Ensure only rows with 'Totals' in Broad Category remain
    if 'Broad category' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['Broad category'].str.contains(r'\bTotal\b', case=False, na=False)]

    # Validate the processed data
    missing_values = df_cleaned.isnull().sum()
    dataset_shape = df_cleaned.shape
    remaining_columns = df_cleaned.columns
    df_cleaned_head = df_cleaned.head()

    return df_cleaned, missing_values, dataset_shape, remaining_columns, df_cleaned_head

# Load the dataset
df_percentage = pd.read_csv("A:/PROJECTS/Data lib/NHS/NHS expenditure per cent of total by budget category and year.csv")
df_expenditure = pd.read_csv("A:/PROJECTS/Data lib/NHS/NHS expenditure by budget category and year.csv")
df_per_head = pd.read_csv("A:/PROJECTS/Data lib/NHS/NHS expenditure per head by budget category and year.csv")


# Execute preprocessing
df_percentage_cleaned, missing_percentage, shape_percentage, cols_percentage, head_percentage = preprocess_nhs_expenditure_percentage(df_percentage)
df_expenditure_cleaned, missing_percentage, shape_percentage, cols_percentage, head_percentage = preprocess_nhs_expenditure_percentage(df_expenditure)
df_per_head_cleaned, missing_percentage, shape_percentage, cols_percentage, head_percentage = preprocess_nhs_expenditure_percentage(df_per_head)

df_per_head_cleaned.to_csv('df_per_head_cleaned.csv', index=False)
df_percentage_cleaned.to_csv('df_percentage_cleaned.csv', index=False)
df_expenditure_cleaned.to_csv('df_expenditure_cleaned.csv', index=False)


df_expenditure = df_expenditure_cleaned
df_percentage = df_percentage_cleaned
df_per_head = df_per_head_cleaned


# Extract expenditure trends for each category
categories = df_expenditure["Broad category"]
years = df_expenditure.columns[1:].astype(str)  # Convert years to string format


# Convert data into time-series format
df_expenditure.set_index("Broad category", inplace=True)
df_percentage.set_index("Broad category", inplace=True)
df_per_head.set_index("Broad category", inplace=True)

df_expenditure = df_expenditure.T  # Transpose to have years as rows
df_percentage = df_percentage.T
df_per_head = df_per_head.T

# Ensure the index is treated as datetime
# Convert financial years to datetime (starting from April 1)
df_expenditure.index = pd.to_datetime(df_expenditure.index.str[:4] + "-04-01")
df_percentage.index = pd.to_datetime(df_percentage.index.str[:4] + "-04-01")
df_per_head.index = pd.to_datetime(df_per_head.index.str[:4] + "-04-01")

df_expenditure = df_expenditure.loc[:'2019']


# Select a category for detailed analysis (e.g., "Total" or "Infectious diseases Total")
category = "Total "
expenditure_series = df_expenditure[category]

# Prepare Data for Prophet
df_prophet = pd.DataFrame({
    "ds": expenditure_series.index,
    "y": expenditure_series.values
})

# --- Step 1: Define Special Events ---
# Define NHS Funding Policy Shifts (Known Change Points)
special_events = pd.DataFrame({
    "ds": pd.to_datetime(["2012-04-01", "2015-04-01", "2018-04-01"]),
    "event": ["Funding Reform 2012", "Budget Increase 2015", "Funding Policy Change 2018"]
})

# Create separate binary indicator columns for each event
for event in special_events["event"].unique():
    df_prophet[event] = 0  # Default to 0
    df_prophet.loc[df_prophet["ds"].isin(special_events["ds"]), event] = 1  # Set to 1 on event dates

# --- Step 2: Define Prophet Model ---
prophet_model = Prophet(
    growth="linear",  
    yearly_seasonality=True,  
    changepoint_prior_scale=0.1  
)

# Add special event regressors
for event in special_events["event"].unique():
    prophet_model.add_regressor(event)

# --- Step 3: Train Prophet Model ---
prophet_model.fit(df_prophet)

# --- Step 4: Generate Forecast for Next 5 Years ---
future = prophet_model.make_future_dataframe(periods=5, freq="Y")

# Ensure regressors exist for forecast period
for event in special_events["event"].unique():
    future[event] = 0  # Set to 0 for future years (since no new policy changes assumed)

# Generate Forecast
forecast = prophet_model.predict(future)

# --- Step 5: Plot Forecast ---
fig = prophet_model.plot(forecast)
plt.title("Prophet Forecast for NHS Expenditure (Excluding COVID-19)")
plt.savefig("prophet_forecast_no_covid.png")

# Save Forecast Data for Upload
forecast.to_csv("prophet_predictions_no_covid.csv", index=False)
 

from prophet.diagnostics import cross_validation, performance_metrics

# Perform Cross-Validation using days (since Prophet doesn't support months/years)
df_cv = cross_validation(
    prophet_model,
    initial="3650 days",  # 10 years = 3650 days
    period="365 days",    # 1 year = 365 days
    horizon="1095 days"   # 3 years = 1095 days
)

# Compute Performance Metrics
df_performance = performance_metrics(df_cv)
print(df_performance)

# Save results for further analysis
df_cv.to_csv("prophet_cross_validation.csv", index=False)
df_performance.to_csv("prophet_performance_metrics.csv", index=False)


from prophet.plot import add_changepoints_to_plot

# Plot Forecast with Changepoints
fig = prophet_model.plot(forecast)
add_changepoints_to_plot(fig.gca(), prophet_model, forecast)
plt.savefig("prophet_changepoints.png")

# Naïve Forecast: Last observed value repeated
naive_forecast = df_cv["y"].shift(1)

# Compute Naïve Forecast MAE for comparison
naive_mae = mean_absolute_error(df_cv["y"], naive_forecast)
prophet_mae = mean_absolute_error(df_cv["y"], df_cv["yhat"])

print(f"Prophet MAE: {prophet_mae}")
print(f"Naïve Forecast MAE: {naive_mae}")


