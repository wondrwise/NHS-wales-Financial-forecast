# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 08:23:38 2025

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
category = "Infectious diseases Total"
expenditure_series = df_expenditure[category]



df_expenditure.head()

# --- 1. Stationarity Check (Dickey-Fuller Test) ---
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is NOT stationary, differencing may be required.")

check_stationarity(expenditure_series)


# --- 2. Variance Stability (Levene’s Test) ---
# Compare pre-2020 and post-2020 periods
pre_covid = expenditure_series.loc[:'2019']
post_covid = expenditure_series.loc['2020':]

stat, p = levene(pre_covid, post_covid)
print("Levene’s Test p-value:", p)
if p < 0.05:
    print("Significant variance difference (increased volatility).")
else:
    print("No significant variance change (stable volatility).")


# --- 3. Autocorrelation Analysis (ACF & PACF) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ACF Plot
axes[0].stem(acf(expenditure_series, nlags=10))
axes[0].set_title("Autocorrelation Function (ACF)")

# PACF Plot
axes[1].stem(pacf(expenditure_series, nlags=5))
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.savefig("acf_pacf_plots.png")  # Save plots for upload

max_lags = min(6, len(expenditure_series) // 2 - 1)  # Ensures nlags is valid

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ACF Plot
axes[0].stem(acf(expenditure_series, nlags=max_lags))
axes[0].set_title("Autocorrelation Function (ACF)")

# PACF Plot
axes[1].stem(pacf(expenditure_series, nlags=max_lags))
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()


# --- 4. Structural Break Detection (Change Point Analysis) ---
import ruptures as rpt


# Convert expenditure series to numpy array (removes index)
signal = expenditure_series.values  

# Apply change point detection using the PELT algorithm
algo = rpt.Pelt(model="rbf").fit(signal)  # 'rbf' detects shifts in mean & variance
breakpoints = algo.predict(pen=10)  # Adjust penalty to control sensitivity

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(expenditure_series, label="NHS Expenditure", marker="o")
for bp in breakpoints[:-1]:  # Exclude the last point (redundant)
    plt.axvline(expenditure_series.index[bp], color="red", linestyle="--", label="Change Point")

plt.title("Structural Breaks in NHS Expenditure")
plt.legend()
plt.show()


# --- Step 1: Apply First-Order Differencing ---
expenditure_series = expenditure_series.diff().dropna()

# ADF Test after differencing
# adf_result = adfuller(expenditure_series)
# print("ADF Statistic (After Differencing):", adf_result[0])
# print("p-value:", adf_result[1])
# if adf_result[1] <= 0.05:
#     print("The time series is now stationary.")
# else:
#     print("Further differencing may be required.")






# --- Step 2: Fit SARIMA Model ---
# Best parameters for SARIMA (Manual tuning or grid search may be needed)
p, d, q = 2, 1, 0  # Based on ACF/PACF
P, D, Q, s = 0, 1, 0, 6  # Seasonal component

sarima_model = SARIMAX(expenditure_series, 
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
sarima_fit = sarima_model.fit()

# Forecast next 5 years
sarima_forecast = sarima_fit.get_forecast(steps=5)
sarima_predicted = sarima_forecast.predicted_mean
sarima_conf_int = sarima_forecast.conf_int()


# Plot SARIMA Forecast
plt.figure(figsize=(10, 5))
plt.plot(expenditure_series, label="Actual", color="blue")
plt.plot(sarima_predicted, label="SARIMA Forecast", color="red", linestyle="dashed")
plt.fill_between(sarima_predicted.index, 
                 sarima_conf_int.iloc[:, 0], 
                 sarima_conf_int.iloc[:, 1], 
                 color='red', alpha=0.2)
plt.title("SARIMA Forecast for NHS Expenditure on Infectious diseases Total")
plt.legend()


import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load cleaned dataset
df_expenditure = pd.read_csv("df_expenditure_cleaned.csv")

# Transpose for time-series format
df_expenditure.set_index("Broad category", inplace=True)
df_expenditure = df_expenditure.T  # Years as index

# Convert index to datetime format
df_expenditure.index = pd.to_datetime(df_expenditure.index, format="%Y-%m")

# Select the category to forecast
# category = "Total"
expenditure_series = df_expenditure[category]

# Define grid search parameters
p = d = q = range(0, 3)  # ARIMA parameters
P = D = Q = range(0, 2)  # Seasonal ARIMA parameters
s = [2, 4, 6, 12]  # Seasonal period options

# Generate all possible parameter combinations
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q, s))

best_aic = float("inf")
best_params = None
best_seasonal_params = None
best_model = None

# Perform grid search
for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = SARIMAX(
                expenditure_series,
                order=param,
                seasonal_order=seasonal_param,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit()
            aic = results.aic  # Get AIC score
            
            # Save best model
            if aic < best_aic:
                best_aic = aic
                best_params = param
                best_seasonal_params = seasonal_param
                best_model = results
            
            print(f"Tested SARIMA{param} x {seasonal_param} - AIC: {aic}")
        except:
            continue

# Print the best model parameters
print("\nBest SARIMA Model:")
print(f"Order: {best_params}")
print(f"Seasonal Order: {best_seasonal_params}")
print(f"Lowest AIC: {best_aic}")

# Save best parameters for upload
with open("best_sarima_params.txt", "w") as f:
    f.write(f"Best Order: {best_params}\n")
    f.write(f"Best Seasonal Order: {best_seasonal_params}\n")
    f.write(f"Lowest AIC: {best_aic}\n")

# Save best model forecast
forecast = best_model.get_forecast(steps=5)
forecast_mean = forecast.predicted_mean
forecast_conf = forecast.conf_int()

# Plot Best SARIMA Forecast
plt.figure(figsize=(10, 5))
plt.plot(expenditure_series, label="Actual", color="blue")
plt.plot(forecast_mean, label="Best SARIMA Forecast", color="red", linestyle="dashed")
plt.fill_between(forecast_mean.index, 
                 forecast_conf.iloc[:, 0], 
                 forecast_conf.iloc[:, 1], 
                 color='red', alpha=0.2)
plt.title("Optimized SARIMA Forecast for NHS Expenditure")
plt.legend()
plt.savefig("best_sarima_forecast.png")

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# Load best SARIMA model
df_expenditure = pd.read_csv("df_expenditure_cleaned.csv")
df_expenditure.set_index("Broad category", inplace=True)
df_expenditure = df_expenditure.T
df_expenditure.index = pd.to_datetime(df_expenditure.index, format="%Y-%m")

# Select the target category
category = "Total"
expenditure_series = df_expenditure[category]

# Fit the optimized SARIMA model
best_sarima_model = sm.tsa.statespace.SARIMAX(
    expenditure_series, 
    order=(0, 1, 0), 
    seasonal_order=(0, 1, 0, 12),
    enforce_stationarity=False, 
    enforce_invertibility=False
).fit()

# Residual Analysis
residuals = best_sarima_model.resid

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residual Plot
axes[0].plot(residuals, label="Residuals", color="blue")
axes[0].axhline(y=0, color='black', linestyle='dashed')
axes[0].set_title("Residual Plot")
axes[0].legend()

# Histogram of Residuals
axes[1].hist(residuals, bins=15, color="blue", alpha=0.7)
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("sarima_residuals.png")

# Ljung-Box Test for autocorrelation
ljung_box_results = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box Test Results:\n", ljung_box_results)

# Out-of-Sample Validation
train_size = int(len(expenditure_series) * 0.8)
train, test = expenditure_series[:train_size], expenditure_series[train_size:]

sarima_model = sm.tsa.statespace.SARIMAX(
    train, 
    order=(0, 1, 0), 
    seasonal_order=(0, 1, 0, 12),
    enforce_stationarity=False, 
    enforce_invertibility=False
).fit()

forecast = sarima_model.get_forecast(steps=len(test))
predicted = forecast.predicted_mean

# Plot Actual vs. Predicted
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label="Actual", color="blue")
plt.plot(test.index, predicted, label="Predicted (SARIMA)", color="red", linestyle="dashed")
plt.fill_between(test.index, 
                 forecast.conf_int().iloc[:, 0], 
                 forecast.conf_int().iloc[:, 1], 
                 color='red', alpha=0.2)
plt.title("SARIMA Model Validation (Actual vs Predicted)")
plt.legend()
plt.savefig("sarima_validation.png")

































# --- Step 3: Fit Prophet Model ---
df_prophet = pd.DataFrame({
    "ds": expenditure_series.index,
    "y": expenditure_series.values
})

# Define Prophet model
prophet_model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1)
prophet_model.fit(df_prophet)

# Create future dates
future = prophet_model.make_future_dataframe(periods=5, freq='Y')

# Forecast using Prophet
forecast = prophet_model.predict(future)

# Plot Prophet Forecast
fig = prophet_model.plot(forecast)
plt.title("Prophet Forecast for NHS Expenditure")
plt.savefig("prophet_forecast.png")

# Save forecasted values for upload
sarima_predicted.to_csv("sarima_predictions.csv")
forecast.to_csv("prophet_predictions.csv")


#
#
#

# exclude covid 19

# Remove COVID-19 period (2019-2022)


# Select the category to forecast (e.g., Total NHS Expenditure)
category = "Total "
expenditure_series = df_expenditure[category]

# --- Step 1: Apply First-Order Differencing (Again) ---
expenditure_diff = expenditure_series.diff().dropna()

# ADF Test after differencing
adf_result = adfuller(expenditure_diff)
print("ADF Statistic (After Differencing, No COVID Data):", adf_result[0])
print("p-value:", adf_result[1])
if adf_result[1] <= 0.05:
    print("The time series is now stationary.")
else:
    print("Further differencing may be required.")

