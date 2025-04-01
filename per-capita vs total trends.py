# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 02:08:18 2025

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
    'expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/NHS expenditure by budget category and year.csv',
    'percentage_of_total': 'A:/PROJECTS/Data lib/NHS/NHS expenditure per cent of total by budget category and year.csv',
    'expenditure_per_head': 'A:/PROJECTS/Data lib/NHS/NHS expenditure per head by budget category and year.csv'
    }

# Load datasets
datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}


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

df_cleaned2, missing_values, dataset_shape, remaining_columns, df_cleaned_head = preprocess_nhs_expenditure_data(datasets["percentage_of_total"])



df_per_head_cleaned = df_cleaned2

df_expenditure_cleaned = df_cleaned


# Step 1: Compare Per Capita vs. Total Expenditure Trends


def plot_per_capita_vs_total(df_total, df_per_head):
    """
    Plots total NHS expenditure alongside per capita expenditure for comparison.
    """
    plt.figure(figsize=(12, 6))

    # Extract years (assume same structure for both datasets)
    years = [int(year.split('-')[0]) for year in df_total.columns[1:]]

    # Extract total expenditure (sum across all categories)
    # total_expenditure = df_total.iloc[:, 1:].sum(axis=0)

    # Extract per capita expenditure (sum across all categories)
    per_capita_expenditure = df_per_head.iloc[:, 1:].sum(axis=0)

    # Plot both trends
    # plt.plot(years, total_expenditure, marker='o', linestyle='-', linewidth=2, label="Total NHS Expenditure (£)")
    plt.plot(years, per_capita_expenditure, marker='s', linestyle='--', linewidth=2, label="Per Capita NHS Expenditure (£)")

    plt.xlabel("Year")
    plt.ylabel("Expenditure (£)")
    plt.title(" Per Capita Expenditure (2009-2023)")
    plt.legend()
    plt.grid(True)

    plt.show()

# Execute comparison plot
plot_per_capita_vs_total(df_expenditure_cleaned, df_per_head_cleaned)



def analyze_per_capita_spending(df_per_head):
    """
    Conducts statistical tests to compare pre-pandemic vs. post-pandemic per capita spending.
    - Paired t-Test to check significant changes in spending levels.
    - Levene's Test to assess variance shifts.
    """

    # Define time periods
    pre_pandemic_years = [str(year) for year in range(2009, 2020)]  # 2009-2019
    post_pandemic_years = [str(year) for year in range(2020, 2024)]  # 2020-2023

    # Extract pre- and post-pandemic per capita spending
    df_pre = df_per_head[pre_pandemic_years].astype(float).sum(axis=0)
    df_post = df_per_head[post_pandemic_years].astype(float).sum(axis=0)

    # Paired t-Test: Significance of Per Capita Spending Change
    t_stat, p_value_t = ttest_rel(df_pre, df_post)

    # Levene's Test: Variance Change in Per Capita Spending
    levene_stat, p_value_levene = levene(df_pre, df_post)

    # Create results dictionary
    results = {
        "Paired t-Test": {"T-Statistic": t_stat, "P-Value": p_value_t},
        "Levene’s Test (Variance)": {"Levene Statistic": levene_stat, "P-Value": p_value_levene}
    }

    return pd.DataFrame.from_dict(results, orient="index")

# Execute statistical tests on per capita expenditure
per_capita_stats = analyze_per_capita_spending(df_per_head_cleaned)

# Display results
print(" Statistical Tests on Per Capita NHS Expenditure:")
print(per_capita_stats)

df_per_head_cleaned.to_csv('df_per_head_cleaned.csv', index=False)




def compute_growth_rates(df_total, df_per_head):
    """
    Computes and visualizes the annual growth rates of total NHS expenditure vs. per capita expenditure.
    """
    # Extract years (assuming both datasets have the same structure)
    years = [str(year) for year in range(2009, 2023)]  # Growth from 2010 to 2023

    # Compute growth rates for total expenditure
    total_expenditure = df_total.iloc[:, 1:].sum(axis=0)  # Summing across all categories
    total_growth = total_expenditure.pct_change() * 100  # Percentage change

    # Compute growth rates for per capita expenditure
    per_capita_expenditure = df_per_head.iloc[:, 1:].sum(axis=0)  # Summing across all categories
    per_capita_growth = per_capita_expenditure.pct_change() * 100  # Percentage change

    # Create a dataframe for visualization
    growth_df = pd.DataFrame({"Year": years, "Total Growth (%)": total_growth.values, "Per Capita Growth (%)": per_capita_growth.values})
    growth_df.set_index("Year", inplace=True)

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(growth_df.index, growth_df["Total Growth (%)"], marker="o", linestyle="-", label="Total NHS Expenditure Growth (%)", linewidth=2)
    plt.plot(growth_df.index, growth_df["Per Capita Growth (%)"], marker="s", linestyle="--", label="Per Capita NHS Expenditure Growth (%)", linewidth=2)
    
    plt.xlabel("Year")
    plt.ylabel("Growth Rate (%)")
    plt.title("Comparison: NHS Total Expenditure Growth vs. Per Capita Spending Growth (2010-2023)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return growth_df

# Execute the growth rate comparison
growth_rate_df = compute_growth_rates(df_expenditure_cleaned, df_per_head_cleaned)

# Display computed growth rates
print(" NHS Expenditure Growth Rates (Total vs. Per Capita):")
print(growth_rate_df)




def compute_spending_volatility(df_per_head):
    """
    Computes and visualizes the variance (volatility) of per capita NHS spending over time.
    """
    # Extract years for analysis
    years = [str(year) for year in range(2009, 2023)]

    # Compute rolling standard deviation to measure volatility
    rolling_volatility = df_per_head[years].astype(float).sum(axis=0).rolling(window=3, center=True).std()

    # Plot volatility trends
    plt.figure(figsize=(12, 6))
    plt.plot(years, rolling_volatility, marker="o", linestyle="-", linewidth=2, color="red", label="Per Capita Spending Volatility")
    
    plt.xlabel("Year")
    plt.ylabel("Standard Deviation (£)")
    plt.title("Volatility of Per Capita NHS Spending Over Time (2010-2023)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return rolling_volatility

# Execute volatility analysis
spending_volatility_df = compute_spending_volatility(df_per_head_cleaned)

# Display computed volatility data
print("Per Capita NHS Spending Volatility:")
print(spending_volatility_df)


#
#
#


def preprocess_nhs_expenditure_percentage(df):
    """
    Prepares NHS expenditure as a percentage of total dataset for analysis.
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

# Execute preprocessing
df_percentage_cleaned, missing_percentage, shape_percentage, cols_percentage, head_percentage = preprocess_nhs_expenditure_percentage(df_percentage)

# Display preprocessing results
print("Dataset Shape:", shape_percentage)
print("Remaining Columns:", cols_percentage)
print("Missing Values:\n", missing_percentage)
print("Preview of Cleaned Data:\n", head_percentage)




def compute_budget_yoy_changes(df_percentage):
    """
    Computes year-over-year (YoY) percentage point changes in NHS expenditure allocation.
    Identifies which categories gained or lost the most as a % of total NHS spending.
    """
    # Extract correct column names from the dataset
    years = [col for col in df_percentage.columns if col not in ["Broad category"]]  
    # Ensuring only valid columns are selected for YoY change calculation

    # Compute year-over-year absolute percentage point change
    df_budget_changes = df_percentage.set_index("Broad category")[years].astype(float).diff(axis=1)

    # Identify biggest gainers and losers
    max_increase = df_budget_changes.max(axis=1).idxmax()  # Category with highest YoY increase
    max_decrease = df_budget_changes.min(axis=1).idxmin()  # Category with highest YoY decrease

    return df_budget_changes, max_increase, max_decrease

# Execute budget shifts analysis
budget_yoy_changes_df, top_gainer, top_loser = compute_budget_yoy_changes(df_percentage_cleaned)

# Display results
print(" Year-over-Year Budget Shifts by Category:")
print(budget_yoy_changes_df.head(10))  # Display first 10 rows

# Print biggest gainers and losers
print(f"\nBiggest Increase in Budget Allocation: {top_gainer}")
print(f" Biggest Decrease in Budget Allocation: {top_loser}")



df_percentage_cleaned.info()



def visualize_budget_shifts(df_budget_changes):
    """
    Generates a heatmap to visualize how NHS budget allocations changed over time.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_budget_changes, cmap="coolwarm", annot=False, linewidths=0.5)

    plt.title("NHS Budget Allocation Changes (Year-over-Year)")
    plt.xlabel("Year")
    plt.ylabel("Broad Category")
    plt.show()

# Execute visualization
visualize_budget_shifts(budget_yoy_changes_df)




def plot_budget_trends(df_percentage, categories):
    """
    Plots NHS budget allocation trends for selected categories over time.
    """
    plt.figure(figsize=(12, 6))

    years = [col for col in df_percentage.columns if col not in ["Broad category"]]  

    for category in categories:
        category_data = df_percentage[df_percentage["Broad category"] == category][years].values.flatten()
        plt.plot(years, category_data, marker="o", linestyle="-", linewidth=2, label=category)

    plt.xlabel("Year")
    plt.ylabel("Percentage of Total NHS Budget")
    plt.title("NHS Budget Allocation Trends for Key Categories (2009-2023)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Execute visualization for top increasing & decreasing categories
plot_budget_trends(df_percentage_cleaned, [top_gainer, top_loser])




def analyze_budget_shift_significance(df_percentage, category):
    """
    Conducts statistical tests to determine if budget shifts for a given category are significant.
    - Paired t-Test: Checks if the mean budget allocation changed significantly pre vs. post-pandemic.
    - Levene’s Test: Assesses whether budget allocation became more volatile over time.
    """

    # Define pre-pandemic and post-pandemic periods
    pre_pandemic_years = [col for col in df_percentage.columns if col not in ["Broad category",'2020-21', '2021-22', '2022-23']]   # 2009-2019
    post_pandemic_years = [col for col in df_percentage.columns if col not in ["Broad category", '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', 
                      '2015-16', '2016-17', '2017-18', '2018-19', '2019-20']]  # 2020-2023

    # Extract category-specific data
    category_data = df_percentage[df_percentage["Broad category"] == category]

    # Ensure data is correctly shaped before applying statistical tests
    if category_data.shape[0] == 0:
        return f"Category {category} not found in dataset."

    # Extract pre- and post-pandemic budget percentages
    pre_pandemic_values = category_data[pre_pandemic_years].values.flatten().astype(float)
    post_pandemic_values = category_data[post_pandemic_years].values.flatten().astype(float)

    # Perform Paired t-Test
    t_stat, p_value_t = ttest_rel(pre_pandemic_values, post_pandemic_values)

    # Perform Levene’s Test for variance
    levene_stat, p_value_levene = levene(pre_pandemic_values, post_pandemic_values)

    # Store results
    results = {
        "Category": category,
        "Paired t-Test": {"T-Statistic": t_stat, "P-Value": p_value_t},
        "Levene’s Test (Variance)": {"Levene Statistic": levene_stat, "P-Value": p_value_levene}
    }

    return pd.DataFrame.from_dict(results, orient="index")

# Execute statistical analysis for the key categories
infectious_disease_stats = analyze_budget_shift_significance(df_percentage_cleaned, top_gainer)
musculo_skeletal_stats = analyze_budget_shift_significance(df_percentage_cleaned, top_loser)

# Display results
print("\n📊 Statistical Tests on NHS Budget Shifts (Infectious Diseases):")
print(infectious_disease_stats)

print("\n📊 Statistical Tests on NHS Budget Shifts (Musculo Skeletal System Problems):")
print(musculo_skeletal_stats)


df_percentage_cleaned.to_csv('df_percentage_cleaned.csv', index=False)



def plot_stacked_area_chart(df_percentage):
    """
    Generates a stacked area chart to visualize NHS expenditure allocation trends over time.
    """
    plt.figure(figsize=(12, 6))
    
    # Extract years and categories
    years = [col for col in df_percentage.columns if col not in ["Broad category"]] 
    categories = df_percentage["Broad category"]

    # Plot stacked area chart
    plt.stackplot(years, df_percentage[years].T, labels=categories, alpha=0.7)

    plt.xlabel("Year")
    plt.ylabel("Percentage of Total NHS Budget")
    plt.title("NHS Spending Trends by Category (2009-2023)")
    plt.legend(loc="upper left", fontsize="small", bbox_to_anchor=(1,1))  # Move legend outside
    plt.grid(True)
    plt.show()

# Execute visualization
plot_stacked_area_chart(df_percentage_cleaned)
