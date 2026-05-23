# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 13:32:43 2026

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
    'Wales expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/NHS expenditure per cent of total by budget category and year.csv',
    
    'Abertawe Bro Morgannwg University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Abertawe Bro Morgannwg University Health Board.csv',
    
    'Aneurin Bevan University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Aneurin Bevan University Health board.csv',
    
    'Betsi Cadwaladr University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Betsi Cadwaladr University Health board.csv',
    
    'Cardiff and vale University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cardiff and vale University Health board.csv',
    
    'Cwm Taf Morgannwg University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cwm Taf Morgannwg University Health board.csv',
    
    'Cwm Taf University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cwm Taf University Health board.csv',
    
    'Hywel Dda University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Hywel Dda University Health board.csv',
    
    'Powys Teaching expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Powys Teaching Health board.csv',
    
    'Swansea Bay expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Swansea Bay Univerity Health board.csv',
    }

# Load datasets
datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}