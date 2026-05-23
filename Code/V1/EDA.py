# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:17:30 2025

@author: edwar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel, levene
from scipy.stats import zscore
import matplotlib.pyplot as plt

for name, df in datasets.items():
    print('Dataset:', name)
    df.info()