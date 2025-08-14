# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 12:59:10 2025

@author: edwar
"""

import re
import logging
from pathlib import Path
import pandas as pd

#logging for info messeges
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# path of all files that need to be cleaned. 
file_paths = {
    'wales_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\NHS expenditure by budget category and year.csv',
    'abertawe_bro_morgannwg_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Abertawe Bro Morgannwg University Health Board.csv',
    'aneurin_bevan_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Aneurin Bevan University Health board.csv',
    'betsi_cadwaladr_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Betsi Cadwaladr University Health board.csv',
    'cardiff_vale_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Cardiff and vale University Health board.csv',
    'cwm_taf_morgannwg_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Cwm Taf Morgannwg University Health board.csv',
    'cwm_taf_university_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Cwm Taf University Health board.csv',
    'hywel_dda_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Hywel Dda University Health board.csv',
    'powys_teaching_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Powys Teaching Health board.csv',
    'swansea_bay_expenditure_by_category': r'A:\PROJECTS\Data lib\NHS\budget category and year\Swansea Bay Univerity Health board.csv',
}

# output of cleaned dataframes
output_dir = Path(r'A:\PROJECTS\Data lib\NHS\budget category and year\refined data')
output_dir.mkdir(parents=True, exist_ok=True)

# define the pattern the years appear used to detect which columns are Year columns.
year_pattern = re.compile(r'^\d{4}-\d{2}$')

# Load datasets
def load_datasets(file_paths: dict[str,str]) ->dict[str, pd.DataFrame]:
    datasets = {}
    for name, path in file_paths.items():
        df = pd.read_csv(path)
        datasets[name] = df
        logging.info(f'Loaded {name}:, shape={df.shape}')
    
    return datasets


# Standardize column names
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
        .str.strip() # removes spaces
        .str.lower() # lowercases
        .str.replace(r"[^\w\s-]", "", regex=True) # removes special characters except underscores/ hyphens
        .str.replace(r"\s+", "_", regex=True) # replaces any space  with single underscore
        )
    # Unify category column name variations
    # unify category column name variations
    if "category_breakdown" not in df.columns and "categorybreakdown" in df.columns:
        df = df.rename(columns={"categorybreakdown": "category_breakdown"})
    if "category_breakdown" not in df.columns and "category_breakdown" in df.columns:
           pass
    elif "category_breakdown" not in df.columns and "category breakdown" in df.columns:
        df = df.rename(columns={"category breakdown": "category_breakdown"})
    return df

datasets = stan(file_paths)  





