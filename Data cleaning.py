# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:03:13 2025

@author: edwar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel, levene
from scipy.stats import zscore
import matplotlib.pyplot as plt


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

#
#

# Data cleaning and reformating

# 
#


# data frame inspection
wales_expenditure = datasets['Wales expenditure_by_category']

wales_info= wales_expenditure.info()
print(wales_info)


for name, df in datasets.items():
    print('Dataset:', name)
    df.info()

    
# Renaming columns ie[2019-20 (4) to 2019-20]

for name, df in datasets.items():
    if '2019-20 (4)' in df.columns:
        df.rename(columns={'2019-20 (4)' : '2019-20'}, inplace=True)

#
#
        
# Droping columns 

drop_columns = [ 'Broadcategory',
    "2020-21LHBprimary", "2020-21LHBsecondary", "2020-21LHBandPHWother",
    "2021-22LHBprimary", "2021-22LHBsecondary", "2021-22LHBandPHWother",
    "2022-23LHBprimary", "2022-23LHBsecondary", "2022-23LHBandPHWother"
    ]

for name, df in datasets.items():
    df.columns = df.columns.str.strip().str.replace(' ','')
    
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
    
#
#

# Combining data frames to represent the shift in Health boards in 2009

# Combining Abertawe Bro with Swansea Bay
# Combining Cwm Taf uni with Cwm Taf Morganmwg

# trailling does not have data from 2019 - 2023
trailing = ['Abertawe Bro Morgannwg University expenditure_by_category', 'Cwm Taf University expenditure_by_category',]

#Leading does not have data from 2009 - 2019
leading = ['Swansea Bay expenditure_by_category', 'Cwm Taf Morgannwg University expenditure_by_category']

# Droping empty colmums in the trailling and Leading groups of data sets
# define what to drop
trailing_drop = ['2019-20','2020-21LHBandPHWtotal','2021-22LHBandPHWtotal','2022-23LHBandPHWtotal']
leading_drop = ['2009-10','2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17','2017-18','2018-19']

# for loop to drop
for name, df in datasets.items():
    if name in trailing:
        df.drop(columns=[col for col in trailing_drop], inplace =True)
        
    elif name in leading:
        df.drop(columns=[col for col in leading_drop], inplace =True)
        

swansea = datasets['Swansea Bay expenditure_by_category']
Abertawe = datasets['Abertawe Bro Morgannwg University expenditure_by_category']


Swansea_Bay_expenditure_by_category = pd.merge(Abertawe, swansea, on='Categorybreakdown', how='inner')

Swansea_Bay_expenditure_by_category.info()

key = 'Swansea Bay expenditure_by_category'
value = Swansea_Bay_expenditure_by_category

datasets[key] = value

Cwm_Taf_Morgannwg_University = datasets['Cwm Taf Morgannwg University expenditure_by_category']
Cwm_Taf_University = datasets['Cwm Taf University expenditure_by_category']

Cwm_Taf_University_expenditure_by_category = pd.merge(Cwm_Taf_University, Cwm_Taf_Morgannwg_University, on='Categorybreakdown', how='inner')

Cwm_Taf_University_expenditure_by_category.info()

key = 'Cwm Taf Morgannwg University expenditure_by_category'
value = Cwm_Taf_University_expenditure_by_category

datasets[key] = value


# Delete Merged redundunt data frames

#
#

# rename (LHBandPHWtotal) data columns 

for name, df in datasets.items():
    if '2020-21LHBandPHWtotal' in df.columns:
        df.rename(columns={'2020-21LHBandPHWtotal' : '2020-21'}, inplace=True)
        

for name, df in datasets.items():
    if '2021-22LHBandPHWtotal' in df.columns:
        df.rename(columns={'2021-22LHBandPHWtotal' : '2021-22'}, inplace=True)
        
for name, df in datasets.items():
    if '2022-23LHBandPHWtotal' in df.columns:
        df.rename(columns={'2022-23LHBandPHWtotal' : '2022-23'}, inplace=True)
        
# remove brackets and numbers form the category breakdown

for name, df in datasets.items():
    if 'Categorybreakdown' in df.columns:
        df['Categorybreakdown'] = (
            df['Categorybreakdown'].astype(str) # Converts to string
            .str.replace(r'\d+', '', regex=True) # Remove numbers
            .str.replace(r'[()]', '', regex=True) # Remove Brackets
            )
        
#
#

# Download cleaned data to be used for next steps

for name, df in datasets.items():
    df.to_csv(f'{name}.csv', index=False)





# EDA

# Mean for all categories 

for name, df in datasets.items():
    # Find numeric columns
    num_cols = df.select_dtypes(include ='number')
    
    if not num_cols.empty:
        # Add mean column
        df['Category_mean'] = num_cols.mean(axis=1)
        print(f"Added 'category_mean' to: {name}")
    else:
        print(f" No numeric cols found in: {name}")
    
        
# plotting mean on histograms for 1 data set.

wales_expenditure = datasets['Wales expenditure_by_category']

wales_expenditure.info()   


# plot histogram
plt.figure(figsize=(8,5))
plt.hist(wales_expenditure['Category_mean'],  edgecolor='black')\

# Add labes and title
plt.xlabel('Category Mean')
plt.ylabel('Frequency')
plt.title('Expenditure category means  - wales')
plt.grid(True)
plt.show()


# plot Horisontal bar chat
# plot Horisontal bar chat for one data set 

categories = wales_expenditure['Categorybreakdown']
means = wales_expenditure['Category_mean']

plt.figure(figsize=(10,10))
plt.barh(categories, means)
plt.xlabel('Mean Expenditure')
plt.title('Mean Wales NHS expenditure by category')
plt.tight_layout()
plt.show()

# plot Horisontal bar chat for all data sets 

for name, df in datasets.items():
    
    # check if required columns exists 
    if 'Categorybreakdown' in df.columns and 'Category_mean' in df.columns:
        
        categories = wales_expenditure['Categorybreakdown']
        means = wales_expenditure['Category_mean']
        
        plt.figure(figsize=(10,10))
        plt.barh(categories, means)
        plt.xlabel('Mean Expenditure')
        plt.title(f'Mean  NHS expenditure by category - {name}')
        plt.tight_layout()
        plt.show()
        
    else:
        print(f'Skipped {name} : Required columns missong')
        
        
        
        
    