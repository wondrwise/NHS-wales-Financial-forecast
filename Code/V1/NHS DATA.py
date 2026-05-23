# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:25:34 2025

@author: edwar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel, levene
from scipy.stats import zscore


# file paths

file_paths = {
    'Wales expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/NHS expenditure by budget category and year.csv',
    'Abertawe Bro Morgannwg University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Abertawe Bro Morgannwg University Health Board.csv',
    'Aneurin Bevan University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Aneurin Bevan University Health board.csv',
    'Betsi Cadwaladr University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Betsi Cadwaladr University Health board.csv',
    'Cardiff and vale University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cardiff and vale University Health board.csv',
    'Cwm Taf Morgannwg University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cwm Taf Morgannwg University Health board.csv',
    'Cwm Taf University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cwm Taf University Health board.csv',
    'Hywel Dda University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Hywel Dda University Health board.csv',
    'Powys Teaching expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Powys Teaching Health board.csv',
    'Swansea Bay expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Swansea Bay Univerity Health board.csv',
    }

# Load datasets
datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Extract column names form each dataset

column_names = {name: set(df.columns) for name, df in datasets.items()}

# Find differences between datasets
expenditure_vs_percentage = column_names["expenditure_by_category"] - column_names["percentage_of_total"]
percentage_vs_expenditure = column_names["percentage_of_total"] - column_names["expenditure_by_category"]

expenditure_vs_per_head = column_names["expenditure_by_category"] - column_names["expenditure_per_head"]
per_head_vs_expenditure = column_names["expenditure_per_head"] - column_names["expenditure_by_category"]

percentage_vs_per_head = column_names["percentage_of_total"] - column_names["expenditure_per_head"]
per_head_vs_percentage = column_names["expenditure_per_head"] - column_names["percentage_of_total"]

# Print column differences
print("Differences between Expenditure and Percentage datasets:")
print("Columns in Expenditure but not in Percentage:", expenditure_vs_percentage)
print("Columns in Percentage but not in Expenditure:", percentage_vs_expenditure)
print()

print("Differences between Expenditure and Per Head datasets:")
print("Columns in Expenditure but not in Per Head:", expenditure_vs_per_head)
print("Columns in Per Head but not in Expenditure:", per_head_vs_expenditure)
print()

print("Differences between Percentage and Per Head datasets:")
print("Columns in Percentage but not in Per Head:", percentage_vs_per_head)
print("Columns in Per Head but not in Percentage:", per_head_vs_percentage)



# preprocessing 

def preprocess_nhs_expenditure_data(df):
    
    # step 1 Drop unnecessary columns 
    
    lhb_columns = [ 'Category breakdown',
    "2020-21 LHB primary", "2020-21 LHB secondary", "2020-21 LHB and PHW other",
    "2021-22 LHB primary", "2021-22 LHB secondary", "2021-22 LHB and PHW other",
    "2022-23 LHB primary", "2022-23 LHB secondary", "2022-23 LHB and PHW other"
    ]
    
    df_cleaned = df.drop(columns = [col for col in lhb_columns if col in df.columns], errors='ignore')
    
    # Step 2: Remove rows where 'Broad category' do not contain 'Totals'
    # Validate available values in broad category
    
    print("Unique Broad Categories BEFORE Filtering:")
    print(df_cleaned['Broad category'].unique())
    
    # Strip whitespaces before filtering
    
    #df_cleaned['Broad category'] = df_cleaned['Broad category'].astype(str).str.strip()
    
    if 'Broad category' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['Broad category']. str.contains(r'\bTotal\b', case=False, na=False)]
    
    # Validate filtering outcome
    print("Unique Broad Categories AFTER Filtering:")
    print(df_cleaned['Broad category'].unique())
    
    # Step 3: Validation checks
    missing_values = df_cleaned.isnull().sum()
    datasets_shape = df_cleaned.shape
    remaining_columns = df_cleaned.columns
    df_cleaned_head = df_cleaned.head()
    
    return df_cleaned2, missing_values, datasets_shape, remaining_columns, df_cleaned_head

# Call the function with the actual DataFrame
df_cleaned, missing_values, dataset_shape, remaining_columns, df_cleaned_head = preprocess_nhs_expenditure_data(datasets["expenditure_by_category"])

df_cleaned2, missing_values, dataset_shape, remaining_columns, df_cleaned_head = preprocess_nhs_expenditure_data(datasets["expenditure_per_head"])


# Print results for validation
print("Cleaned Dataset Shape:", dataset_shape)
print("Remaining Columns:", remaining_columns)
print("Missing Values:\n", missing_values)
print("Preview of Cleaned Data:\n", df_cleaned_head)

df_cleaned.info()

# Melt the data into long format
df_long = df_cleaned.melt(id_vars=['Broad category'], var_name='Year', value_name='Expenditure')

# Convert 'Year' column to match GDP deflator format (keep only the starting year)
df_long['Year'] = df_long['Year'].apply(lambda x: int(x.split('-')[0]))

# GDP deflator values from NHS Digital
gdp_deflator_data = {
    'Year': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'GDP Deflator': [70.799, 72.133, 73.405, 74.752, 76.189, 77.116, 77.671, 79.438, 80.682, 82.384, 84.329, 88.921, 88.192, 94.134]
}

# Convert to DataFrame
df_deflator = pd.DataFrame(gdp_deflator_data)

# Get the base year deflator (2022) since our dataset ends in 2022-23
base_year = 2022
base_deflator = df_deflator.loc[df_deflator['Year'] == base_year, 'GDP Deflator'].values[0]

# Compute the adjustment factor based on 2022 prices
df_deflator['Adjustment Factor'] = base_deflator / df_deflator['GDP Deflator']

# Merge NHS expenditure data with GDP deflator data
df_long = df_long.merge(df_deflator, on='Year', how='left')

# Apply inflation adjustment
df_long['Expenditure_Adjusted'] = df_long['Expenditure'] * df_long['Adjustment Factor']

# Pivot back to wide format
df_adjusted = df_long.pivot(index='Broad category', columns='Year', values='Expenditure_Adjusted').reset_index()

# Rename columns to match original format
df_adjusted.columns = ['Broad category'] + [f'{year}-{str(year+1)[-2:]}' for year in range(2009, 2023)]

df_cleaned = df_adjusted


#
#
#
#
#

def plot_total_expenditure_trend(df):
    
    """
    Plots the trend of NHS total expenditure from 2009-10 to 2022-23.
    """
    
    # Extract total row 
    total_expenditure = df.iloc[0, 2:] # Excluding 'Broad category' and 'Category breakdown' columns
    
    # convert index (years) to integers
    years = [int(year.split('-')[0]) for year in total_expenditure.index]
    
    # plot the trend
    plt.figure(figsize=(12,6))
    plt.plot(years, total_expenditure, marker='o', linestyle='-', color='b', linewidth=2)
    plt.xlabel('Year')
    plt.ylabel('Total Expenditure (1,000,000)')
    plt.title('NHS Total Expenditure Trend (2009-10 to 2022-23)')
    plt.grid(True)
    
    # show plot
    
    plt.show()

    return

plot_total_expenditure_trend(df_cleaned)

#
#
#
#



def plot_individual_category_expenditure_trends(df):
    """
    Plots separate expenditure trends for each NHS category from 2009-10 to 2022-23.
    """

    # Exclude the first row (Overall total expenditure)
    category_data = df.iloc[1:, 2:]  # Exclude 'Broad category' and 'Category breakdown' columns
    
    # Get category labels
    categories = df.iloc[1:, 0].values  # Broad category names
    
    # Convert index (years) to integers
    years = [int(year.split('-')[0]) for year in category_data.columns]
    
    # Loop through each category and create separate plots
    for i, category in enumerate(categories):
        plt.figure(figsize=(10, 5))
        plt.plot(years, category_data.iloc[i], marker='o', linestyle='-', linewidth=2, label=category, color='b')

        # Formatting
        plt.xlabel('Year')
        plt.ylabel('Expenditure (1000)')
        plt.title(f'NHS Expenditure Trend: {category} (2009-10 to 2022-23)')
        plt.legend()
        plt.grid(True)
        
        # Show plot
        plt.show()

# Call function
plot_individual_category_expenditure_trends(df_cleaned)


#
#
#
#
#


def calculate_expenditure_statistics(df):
    """
    Calculates variance, standard deviation, and growth rate in expenditure for each category over time.
    
    Returns:
        - A sorted DataFrame of categories with highest variance, standard deviation, and growth rate.
    """
    
    category_stats = []

    # Iterate over each category (excluding the total row)
    for i in range(1, df.shape[0]):  # Skip first row (Total expenditure)
        category = df.iloc[i, 0]  # Broad category name
        expenditure_values = df.iloc[i, 2:].values  # Expenditure over years
        
        # Compute variance and standard deviation
        variance = np.var(expenditure_values, ddof=1)  # Sample variance
        std_dev = np.sqrt(variance)  # Standard deviation
        
        # Compute growth rate (percentage increase from first to last year)
        first_year_expenditure = expenditure_values[0]
        last_year_expenditure = expenditure_values[-1]
        
        if first_year_expenditure > 0:  # Avoid division by zero
            growth_rate = ((last_year_expenditure - first_year_expenditure) / first_year_expenditure) * 100
        else:
            growth_rate = np.nan  # Handle cases where the first value is zero
        
        # Append results as a tuple
        category_stats.append((category, variance, std_dev, growth_rate))
    
    # Convert results into a sorted DataFrame
    stats_df = pd.DataFrame(category_stats, columns=['Category', 'Variance', 'Standard Deviation', 'Growth Rate'])
    
    # Sort by variance in descending order
    stats_df = stats_df.sort_values(by='Variance', ascending=False)

    return stats_df

# Compute expenditure statistics
category_stats_df = calculate_expenditure_statistics(df_cleaned)

# Display top 10 categories with highest variance
print(category_stats_df.head(10))  
    
category_stats_df.to_csv('NHS_Expenditure_Statistics.csv', index=False)


df_stats = category_stats_df

# Set index to category names
df_stats.set_index("Category", inplace=True)

# Visualization 1: Heatmap of Variance & Growth Rate
plt.figure(figsize=(12, 6))
sns.heatmap(df_stats[["Variance", "Growth Rate"]], annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Variance & Growth Rate Heatmap (NHS Expenditure Categories)")
plt.show()

# Visualization 2: Bar Chart for Growth Rate
plt.figure(figsize=(12, 6))
df_stats["Growth Rate"].sort_values(ascending=False).plot(kind="bar", color="b", alpha=0.7)
plt.ylabel("Growth Rate (%)")
plt.title("Top NHS Categories by Growth Rate")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#
#
#
#
#


# Pre vs. Post-Pandemic Comparison

  
def compare_pre_post_pandemic(df):
    """
    Splits data into Pre-Pandemic (2009-2019) and Post-Pandemic (2020-2023),
    then performs:
    1. Paired t-Test (to check if spending significantly changed)
    2. Levene's Test (to check if variance significantly changed)
    """

    # Define correct column names for pre- and post-pandemic periods
    pre_pandemic_years = [f"{year}-{str(year+1)[-2:]}" for year in range(2009, 2020)]  # 2009-10 to 2019-20
    post_pandemic_years = [f"{year}-{str(year+1)[-2:]}" for year in range(2020, 2024)]  # 2020-21 to 2022-23

    # Check if columns exist in the DataFrame
    available_pre_years = [col for col in pre_pandemic_years if col in df.columns]
    available_post_years = [col for col in post_pandemic_years if col in df.columns]

    if not available_pre_years or not available_post_years:
        print("Error: Some year columns are missing in the dataset.")
        print("Available columns:", df.columns.tolist())
        print("Missing columns:", [col for col in pre_pandemic_years + post_pandemic_years if col not in df.columns])
        return None, None

    # Extract data for Pre and Post-Pandemic periods
    df_pre = df[available_pre_years].astype(float)
    df_post = df[available_post_years].astype(float)
    


    # Check if data is empty
    if df_pre.empty or df_post.empty:
        print("Error: Pre-pandemic or post-pandemic data is empty.")
        return None, None

    # Paired t-Test (significance of spending changes)
    ttest_results = {}
    for idx, category in enumerate(df.index):
        try:
            t_stat, p_value = ttest_rel(df_pre.iloc[idx], df_post.iloc[idx])
            ttest_results[category] = {'T-Statistic': t_stat, 'P-Value': p_value}
        except ValueError:
            ttest_results[category] = {'T-Statistic': np.nan, 'P-Value': np.nan}  # Handle missing data

    # Convert results to dataframe
    ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index')

    # Levene’s Test (Variance changes)
    levene_results = {}
    for idx, category in enumerate(df.index):
        try:
            stat, p_val = levene(df_pre.iloc[idx], df_post.iloc[idx])
            levene_results[category] = {'Levene Statistic': stat, 'P-Value': p_val}
        except ValueError:
            levene_results[category] = {'Levene Statistic': np.nan, 'P-Value': np.nan}  # Handle missing data

    # Convert results to dataframe
    levene_df = pd.DataFrame.from_dict(levene_results, orient='index')

    return ttest_df, levene_df

# Execute pre vs. post pandemic tests
ttest_df, levene_df = compare_pre_post_pandemic(df_cleaned.set_index("Broad category"))



df_cleaned.to_csv('df_cleaned.csv', index=False)

levene_df.to_csv('levene_df.csv', index=False)


# Load the test results
ttest_results = pd.read_csv("Paired_t-Test_Results.csv", index_col=0)
levene_results = pd.read_csv("Levene_s_Test_Results.csv", index_col=0)

ttest_results= ttest_results.drop('Category Breakdown')

# Identify significant spending changes (Paired t-Test)
significant_spending_changes = ttest_results[ttest_results["p-Value"] < 0.05].sort_values(by="p-Value")

# Identify significant variance changes (Levene's Test)
significant_variance_changes = levene_results[levene_results["p-Value"] < 0.05].sort_values(by="p-Value")

# Display key findings
print(" Significant Spending Changes (Paired t-Test):")
print(significant_spending_changes.head(10))

print("\n Significant Variance Changes (Levene’s Test):")
print(significant_variance_changes.head(10))


#
#
#
#
#

# correlation Analysis

# Ensure numeric data for correlation calculation

# Select only numerical expenditure data (years), keeping Broad category as index
df_corr = df_cleaned.set_index("Broad category").iloc[:, 1:].astype(float)

# Transpose to compute correlation across categories instead of years
correlation_matrix = df_corr.T.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of NHS Expenditure Categories")
plt.show()


#
#
#
#

# Anomaly Detection Using Z-Scores


df_anomaly = df_cleaned.set_index('Broad category').iloc[:,1:].astype(float)

# Compute Z-scores
z_scores = df_anomaly.apply(zscore, axis=1)

threshold = 2.5
anomalies_detected = {}
for idx, category in enumerate(df_anomaly.index):
    anomaly_years = z_scores.columns[(z_scores.iloc[idx].abs() > threshold)].tolist()
    if anomaly_years:
        anomalies_detected[category] = anomaly_years

# Convert anomalies to DataFrame
anomalies_df = pd.DataFrame(list(anomalies_detected.items()), columns=["Category", "Anomaly Years"])

# Display detected anomalies
print(" Detected Anomalies in NHS Expenditure:")
print(anomalies_df.head(10))

#
#
#
#
#

# Comparing NHS Categories Over Time


# Convert numeric columns for trend analysis
df_trends = df_cleaned.set_index("Broad category").iloc[:, 1:].astype(float)

# Define key categories for comparison
key_categories = [
    "Infectious diseases Total", 
    "Mental health problems Total", 
    "Circulation problems Total", 
    "Cancers & tumours Total", 
    "Respiratory problems Total"
]

# Extract relevant data for selected categories
df_selected = df_trends.loc[key_categories]

# Convert column names to years
years = [int(year.split('-')[0]) for year in df_selected.columns]

# Plot category trends
plt.figure(figsize=(12, 6))
for category in key_categories:
    plt.plot(years, df_selected.loc[category], marker='o', linestyle='-', linewidth=2, label=category)

plt.xlabel("Year")
plt.ylabel("Expenditure (£)")
plt.title("NHS Expenditure Trends for Selected Categories (2009-2023)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()



