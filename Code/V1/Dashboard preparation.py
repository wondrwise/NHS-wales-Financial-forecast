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
import re
from datetime import date




# file paths



file_paths = {
    
# Load datsets expenditure_by_category

    'Wales expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Wales expenditure_by_category.csv',
    
    'Aneurin Bevan University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Aneurin Bevan University expenditure_by_category.csv',
    
    'Betsi Cadwaladr University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Betsi Cadwaladr University expenditure_by_category.csv',
    
    'Cardiff and vale University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Cardiff and vale University expenditure_by_category.csv',
    
    'Cwm Taf Morgannwg University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Cwm Taf Morgannwg University expenditure_by_category.csv',
    
    'Hywel Dda University expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Hywel Dda University expenditure_by_category.csv',
    
    'Powys Teaching expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Powys Teaching expenditure_by_category.csv',
    
    'Swansea Bay expenditure_by_category': 'A:/PROJECTS/Data lib/NHS/budget category and year/Cleaned/Swansea Bay expenditure_by_category.csv',
    
# Load datsets per cent of total by budget category and year

    'Wales expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Wales expenditure_by_category.csv',
    
    'Aneurin Bevan University expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Aneurin Bevan University expenditure_by_category.csv',
    
    'Betsi Cadwaladr University expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Betsi Cadwaladr University expenditure_by_category.csv',
    
    'Cardiff and vale University expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Cardiff and vale University expenditure_by_category.csv',
    
    'Cwm Taf Morgannwg University expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Cwm Taf Morgannwg University expenditure_by_category.csv',
    
    'Hywel Dda University expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Hywel Dda University expenditure_by_category.csv',
    
    'Powys Teaching expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Powys Teaching expenditure_by_category.csv',
    
    'Swansea Bay expenditure per cent of total': 'A:/PROJECTS/Data lib/NHS/per cent of total by budget category and year/Cleaned/Swansea Bay expenditure_by_category.csv',

# Load datasets expenditure per head by budget category and year

    'Wales expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Wales expenditure_by_category.csv',
    
    'Aneurin Bevan University expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Aneurin Bevan University expenditure_by_category.csv',
    
    'Betsi Cadwaladr University expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Betsi Cadwaladr University expenditure_by_category.csv',
    
    'Cardiff and vale University expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Cardiff and vale University expenditure_by_category.csv',
    
    'Cwm Taf Morgannwg University expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Cwm Taf Morgannwg University expenditure_by_category.csv',
    
    'Hywel Dda University expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Hywel Dda University expenditure_by_category.csv',
    
    'Powys Teaching expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Powys Teaching expenditure_by_category.csv',
    
    'Swansea Bay expenditure per head': 'A:/PROJECTS/Data lib/NHS/expenditure per head by budget category and year/Cleaned/Swansea Bay expenditure_by_category.csv',
    }

# Load datasets


datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}



# Create the category mapping table from proposed groupings

# Step 1.1.1 — Build DimCategory from a hard-coded mapping (easy to audit and edit)

category_rows = [
    {"Category breakdown": "total", "ServiceArea": "All", "HasBreakdown": "Yes"},  # your overall total row

    {"Category breakdown": "infectious_diseases_total", "ServiceArea": "Infectious", "HasBreakdown": "No"},
    {"Category breakdown": "cancers_tumours_total", "ServiceArea": "Cancer", "HasBreakdown": "No"},
    {"Category breakdown": "blood_disorders_total", "ServiceArea": "Blood", "HasBreakdown": "No"},

    {"Category breakdown": "endocrine_nutritional_metabolic_problems_total", "ServiceArea": "Metabolic", "HasBreakdown": "Yes"},
    {"Category breakdown": "diabetes", "ServiceArea": "Metabolic", "HasBreakdown": "No"},
    {"Category breakdown": "other_endocrine_nutritional_and_metabolic_problems", "ServiceArea": "Metabolic", "HasBreakdown": "No"},

    {"Category breakdown": "mental_health_problems_total", "ServiceArea": "Mental health", "HasBreakdown": "Yes"},
    {"Category breakdown": "general_mental_illness", "ServiceArea": "Mental health", "HasBreakdown": "No"},
    {"Category breakdown": "elderly_mental_illness", "ServiceArea": "Mental health", "HasBreakdown": "No"},
    {"Category breakdown": "child_adolescent_mental_health_services", "ServiceArea": "Mental health", "HasBreakdown": "No"},
    {"Category breakdown": "other_mental_health_problems", "ServiceArea": "Mental health", "HasBreakdown": "No"},

    {"Category breakdown": "learning_disability_problems_total", "ServiceArea": "Learning disability", "HasBreakdown": "No"},
    {"Category breakdown": "neurological_system_problems_total", "ServiceArea": "Neurological", "HasBreakdown": "No"},
    {"Category breakdown": "eyevision_problems_total", "ServiceArea": "Eye vision", "HasBreakdown": "No"},
    {"Category breakdown": "hearing_problems_total", "ServiceArea": "Hearing", "HasBreakdown": "No"},

    {"Category breakdown": "circulation_problems_total", "ServiceArea": "Circulatory", "HasBreakdown": "Yes"},
    {"Category breakdown": "coronary_heart_disease", "ServiceArea": "Circulatory", "HasBreakdown": "No"},
    {"Category breakdown": "cerebrovascular_disease", "ServiceArea": "Circulatory", "HasBreakdown": "No"},
    {"Category breakdown": "other_problems_of_circulation", "ServiceArea": "Circulatory", "HasBreakdown": "No"},

    {"Category breakdown": "respiratory_problems_total", "ServiceArea": "Respiratory", "HasBreakdown": "No"},
    {"Category breakdown": "dental_problems_total", "ServiceArea": "Dental", "HasBreakdown": "No"},
    {"Category breakdown": "gastro_intestinal_problems_total", "ServiceArea": "Intestinal", "HasBreakdown": "No"},
    {"Category breakdown": "skin_problems_total", "ServiceArea": "Skin", "HasBreakdown": "No"},
    {"Category breakdown": "musculo_skeletal_system_problems_exc_trauma_total", "ServiceArea": "Musculo skeletal", "HasBreakdown": "No"},
    {"Category breakdown": "trauma_injuries_inc_burns_total", "ServiceArea": "Trauma injuries", "HasBreakdown": "No"},

    {"Category breakdown": "genito_urinary_system_disorders_exc_infertility_total", "ServiceArea": "Genito urinary", "HasBreakdown": "Yes"},
    {"Category breakdown": "genital_tract_problems", "ServiceArea": "Genito urinary", "HasBreakdown": "No"},
    {"Category breakdown": "renal_problems", "ServiceArea": "Genito urinary", "HasBreakdown": "No"},
    {"Category breakdown": "chronic_renal_failure", "ServiceArea": "Genito urinary", "HasBreakdown": "No"},
    {"Category breakdown": "sexually_transmitted_infections", "ServiceArea": "Genito urinary", "HasBreakdown": "No"},
    {"Category breakdown": "other_problems_of_the_genito_urinary_system", "ServiceArea": "Genito urinary", "HasBreakdown": "No"},

    {"Category breakdown": "maternity_reproductive_health_total", "ServiceArea": "Reproductive", "HasBreakdown": "No"},
    {"Category breakdown": "neonates_total", "ServiceArea": "Neonatal", "HasBreakdown": "No"},
    {"Category breakdown": "poisoning_total", "ServiceArea": "Poisons", "HasBreakdown": "No"},
    {"Category breakdown": "healthy_individuals_includes_screening_total", "ServiceArea": "Screening", "HasBreakdown": "No"},
    {"Category breakdown": "social_care_needs_total", "ServiceArea": "Social care", "HasBreakdown": "No"},

    {"Category breakdown": "other_total", "ServiceArea": "Other", "HasBreakdown": "Yes"},
    {"Category breakdown": "general_medical_services", "ServiceArea": "Other", "HasBreakdown": "No"},
    {"Category breakdown": "open_access", "ServiceArea": "Other", "HasBreakdown": "No"},
    {"Category breakdown": "continuing_care", "ServiceArea": "Other", "HasBreakdown": "No"},
    {"Category breakdown": "other_phw_functions", "ServiceArea": "Other", "HasBreakdown": "No"},
    {"Category breakdown": "other_expenditure", "ServiceArea": "Other", "HasBreakdown": "No"},
]

dim_category = pd.DataFrame(category_rows)

# CategoryType rule: totals end with _total OR are the overall 'total'
dim_category["CategoryType"] = np.where(
    (dim_category["Category breakdown"].str.endswith("_total")) | (dim_category["Category breakdown"] == "total"),
    "Total",
    "Subcategory"
)

# NotesFlag: flag known footnote-sensitive categories
dim_category["NotesFlag"] = np.where(
    dim_category["Category breakdown"].isin(["other_problems_of_circulation"]),
    1,
    0
)

dim_category.head()



# Validate category dimension quality (debug checks)


# Step 1.1.2 — Validation checks for DimCategory

assert dim_category["Category breakdown"].isna().sum() == 0, "Missing Category breakdown values"
assert dim_category["Category breakdown"].duplicated().sum() == 0, "Duplicate Category breakdown keys detected"

print("DimCategory rows:", len(dim_category))
print(dim_category["CategoryType"].value_counts())
print("NotesFlag count:", dim_category["NotesFlag"].sum())


# Check coverage against real data (very important)

# Step 1.1.3 — Compare DimCategory keys vs observed categories in one dataset

sample_name = list(datasets.keys())[7]
sample_df = datasets[sample_name]

likely_cols = [c for c in sample_df.columns if 'category' in c.lower()]
print("Likely category columns:", likely_cols)

CATEGORY_COL = "Category breakdown"  # change if needed

#
#

observed_categories = set(sample_df[CATEGORY_COL].astype(str).str.strip().unique())
mapped_categories = set(dim_category["Category breakdown"].astype(str).str.strip().unique())

missing_in_mapping = sorted(observed_categories - mapped_categories)
extra_in_mapping = sorted(mapped_categories - observed_categories)

print("Categories in data but missing in mapping:", missing_in_mapping[:20], "..." if len(missing_in_mapping) > 20 else "")
print("Categories in mapping but not in data:", extra_in_mapping[:20], "..." if len(extra_in_mapping) > 20 else "")
print("Missing count:", len(missing_in_mapping))
print("Extra count:", len(extra_in_mapping))

#
#

# Phase 1.2 — Financial Year table (DimFinancialYear)

# Identify FY columns from one dataset (debug)


# Step 1.2.0 — Identify year columns
year_pattern = re.compile(r"^\d{4}-\d{2}$")

year_cols = [c for c in sample_df.columns if year_pattern.match(str(c).strip())]
print("Detected financial year columns:", year_cols)


# Build DimFinancialYear


# Step 1.2.1 — Build DimFinancialYear
fy_labels = sorted(year_cols, key=lambda x: int(x[:4]))

dim_fy = pd.DataFrame({"FinancialYear": fy_labels})
dim_fy["FY_StartYear"] = dim_fy["FinancialYear"].str.slice(0, 4).astype(int)
dim_fy["FY_EndYear"] = dim_fy["FY_StartYear"] + 1

# UK financial year runs 1 April to 31 March
dim_fy["FY_StartDate"] = dim_fy["FY_StartYear"].apply(lambda y: date(y, 4, 1))
dim_fy["FY_EndDate"] = dim_fy["FY_EndYear"].apply(lambda y: date(y, 3, 31))

# Sort key (Power BI uses this to sort labels correctly)
dim_fy["FY_Order"] = dim_fy["FY_StartYear"]

dim_fy.head()


# Validation Checks

# Step 1.2.2 — Validations
assert dim_fy["FinancialYear"].duplicated().sum() == 0, "Duplicate FY labels detected"
assert dim_fy["FY_Order"].is_monotonic_increasing, "FY_Order is not increasing"

print("DimFinancialYear rows:", len(dim_fy))
print(dim_fy.tail())

#
#

# Health Board table (DimHealthBoard), including Population
# Extract HealthBoard names from dataset keys (debug)

# Step 1.3.0 — Extract unique HealthBoards from dataset keys

def extract_health_board(dataset_key: str) -> str:
    # remove common suffix patterns from your keys
    key = dataset_key
    key = key.replace(" expenditure_by_category", "")
    key = key.replace(" expenditure per cent of total", "")
    key = key.replace(" expenditure per head", "")
    return key.strip()

health_boards = sorted({extract_health_board(k) for k in datasets.keys()})
health_boards_df = pd.DataFrame({"HealthBoard": health_boards})

print("Detected HealthBoards:")
print(health_boards_df)


# Add BoardType + RegionGroup (editable mapping)

# Step 1.3.1 — BoardType + RegionGroup mapping (edit if you want different labels)

region_map = {
    "Betsi Cadwaladr University": "North Wales",
    "Powys Teaching": "Mid Wales",
    "Hywel Dda University": "West Wales",
    "Swansea Bay": "South West Wales",
    "Cwm Taf Morgannwg University": "South Central Wales",
    "Cardiff and vale University": "Cardiff & Vale",
    "Aneurin Bevan University": "South East Wales",
    "Wales": "Wales (National)"
}

board_type_map = {
    "Wales": "National aggregate"
    # everything else defaults to Local Health Board
}

dim_hb = health_boards_df.copy()
dim_hb["RegionGroup"] = dim_hb["HealthBoard"].map(region_map).fillna("Unknown")
dim_hb["BoardType"] = dim_hb["HealthBoard"].map(board_type_map).fillna("Local Health Board")

dim_hb


# population 

# Load and inspect population file you uploaded

# Step 1.3.2 — Load population estimates file
pop_path = "A:/PROJECTS/Data lib/NHS/Population estimates by local health boards and year.csv"
pop_raw = pd.read_csv(pop_path)

print(pop_raw.shape)
print(pop_raw.columns)
pop_raw.head()

# Standardise population file to a clean long format

year_cols = [c for c in pop_raw.columns if re.fullmatch(r"\d{4}", str(c).strip())]
print("Detected year columns:", year_cols)


# Step 1.3.4 — Melt wide years into a long table
HB_COL = "local health boards"   # based on your screenshot
AREA_COL = "Area code"           # based on your screenshot

pop_long = pop_raw.melt(
    id_vars=[HB_COL, AREA_COL],
    value_vars=year_cols,
    var_name="Year",
    value_name="Population"
)

# Clean and types
pop_long["HealthBoardRaw"] = pop_long[HB_COL].astype(str).str.strip()
pop_long["AreaCode"] = pop_long[AREA_COL].astype(str).str.strip()
pop_long["Year"] = pop_long["Year"].astype(int)
pop_long["Population"] = pd.to_numeric(pop_long["Population"], errors="coerce")

# Keep only clean cols
pop_long = pop_long[["HealthBoardRaw", "AreaCode", "Year", "Population"]].copy()

print(pop_long.shape)
print(pop_long.isna().sum())
pop_long.head()


# Take the latest year per board (for banding)

# Latest population per board

pop_latest = (
    pop_long.dropna(subset=["Population"])
            .sort_values(["HealthBoardRaw", "Year"])
            .groupby("HealthBoardRaw", as_index=False)
            .tail(1)
            .rename(columns={"Year": "PopulationYear"})
)

pop_latest.head()


# Map population board names to your finance board names

# Step 1.3.6 — Map population naming -> finance naming
pop_to_finance_map = {
    "Wales": "Wales",
    "Betsi Cadwaladr University Health Board": "Betsi Cadwaladr University",
    "Powys Teaching Health Board": "Powys Teaching",
    "Hywel Dda University Health Board": "Hywel Dda University",
    "Swansea Bay University Health Board": "Swansea Bay",
    "Cwm Taf Morgannwg University Health Board": "Cwm Taf Morgannwg University",
    "Cardiff and Vale University Health Board": "Cardiff and vale University",
    "Aneurin Bevan University Health Board": "Aneurin Bevan University"
}

pop_latest["HealthBoard"] = pop_latest["HealthBoardRaw"].map(pop_to_finance_map)

unmapped = sorted(pop_latest.loc[pop_latest["HealthBoard"].isna(), "HealthBoardRaw"].unique())
print("Unmapped population board names:", unmapped)

# Drop unmapped for now (we fix mapping if needed)
pop_latest_mapped = pop_latest.dropna(subset=["HealthBoard"]).copy()
pop_latest_mapped.head()


# Step 1.3.7 — Population bands (Small/Medium/Large) using tertiles
pop_for_banding = pop_latest_mapped[pop_latest_mapped["HealthBoard"] != "Wales"].copy()

pop_for_banding["PopulationBand"] = pd.qcut(
    pop_for_banding["Population"],
    q=3,
    labels=["Small", "Medium", "Large"]
)

pop_for_banding[["HealthBoard", "PopulationYear", "Population", "PopulationBand"]].sort_values("Population")


# Step 1.3.8 — Merge population into DimHealthBoard
dim_hb = dim_hb.merge(
    pop_for_banding[["HealthBoard", "PopulationYear", "Population", "PopulationBand"]],
    on="HealthBoard",
    how="left"
)

print(dim_hb.isna().sum())
dim_hb.sort_values("HealthBoard")


dim_category.to_csv('dim_category.csv', index=False)
dim_fy.to_csv('dim_fy.csv', index=False)
dim_hb.to_csv('dim_hb.csv', index=False)









# Phase 2 — Integrate the 9 health board datasets into one reporting layer

# Detect “FinancialYear” value columns

FY_LABEL_PATTERN = re.compile(r"\b(\d{4}-\d{2})\b")   # finds 2009-10 etc.

def get_financial_year_value_columns(df: pd.DataFrame) -> list:
    cols = list(df.columns)

    # Case A: clean year-only columns exist
    year_only = [c for c in cols if re.fullmatch(r"\d{4}-\d{2}", str(c).strip())]
    if year_only:
        return year_only

    # Case B: component-style columns exist (we keep only TOTAL component)
    # We keep columns that contain an FY label AND contain "LHB and PHW total" (case-insensitive)
    total_component_cols = []
    for c in cols:
        c_str = str(c)
        if FY_LABEL_PATTERN.search(c_str) and ("lhb and phw total" in c_str.lower()):
            total_component_cols.append(c)

    return total_component_cols



# Helper: Convert wide → long (FinancialYear + Value)


def wide_to_long_finance(
    df: pd.DataFrame,
    dataset_key: str,
    metric_type: str
) -> pd.DataFrame:
    if CATEGORY_COL not in df.columns:
        raise ValueError(f"Expected category column '{CATEGORY_COL}' not found in {dataset_key}. "
                         f"Available columns: {list(df.columns)}")

    value_cols = get_financial_year_value_columns(df)

    if not value_cols:
        raise ValueError(
            f"No financial year value columns detected for {dataset_key}. "
            f"Check column naming for years or 'LHB and PHW total'."
        )

    tmp = df[[CATEGORY_COL] + value_cols].copy()

    long_df = tmp.melt(
        id_vars=[CATEGORY_COL],
        value_vars=value_cols,
        var_name="YearColumn",
        value_name="Value"
    )

    # Extract FY label (works for both year-only and component-style columns)
    long_df["FinancialYear"] = long_df["YearColumn"].astype(str).str.extract(FY_LABEL_PATTERN)[0]

    # Add HealthBoard and MetricType
    long_df["HealthBoard"] = extract_health_board(dataset_key)
    long_df["MetricType"] = metric_type

    # Clean types
    long_df[CATEGORY_COL] = long_df[CATEGORY_COL].astype(str).str.strip()
    long_df["FinancialYear"] = long_df["FinancialYear"].astype(str).str.strip()
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")

    # Keep only the canonical columns
    long_df = long_df[["HealthBoard", "FinancialYear", CATEGORY_COL, "MetricType", "Value"]].copy()

    return long_df


# Quick validation at target grain

def validate_fact_table(fact: pd.DataFrame, name: str) -> None:
    print(f"\n--- Validation: {name} ---")
    print("Shape:", fact.shape)
    print("Nulls:\n", fact.isna().sum())

    # duplicates at grain
    dup_count = fact.duplicated(subset=["HealthBoard", "FinancialYear", CATEGORY_COL, "MetricType"]).sum()
    print("Duplicate grain rows:", dup_count)

    # FY sanity: show a few unique FYs
    print("Sample FinancialYears:", sorted(fact["FinancialYear"].dropna().unique())[:10])
    print("Sample HealthBoards:", sorted(fact["HealthBoard"].dropna().unique()))



# Build Fact_Expenditure

# Select only the “expenditure_by_category” datasets

expenditure_keys = [k for k in datasets.keys() if k.endswith("expenditure_by_category")]
print("Expenditure datasets found:", len(expenditure_keys))
print(expenditure_keys)

# Convert each to long and stack

fact_expenditure_parts = []

for k in expenditure_keys:
    df = datasets[k]
    long_df = wide_to_long_finance(df=df, dataset_key=k, metric_type="Expenditure")
    fact_expenditure_parts.append(long_df)

fact_expenditure = pd.concat(fact_expenditure_parts, ignore_index=True)

validate_fact_table(fact_expenditure, "Fact_Expenditure")
fact_expenditure.head()


# Build Fact_PerHead

perhead_keys = [k for k in datasets.keys() if k.endswith("expenditure per head")]
print("PerHead datasets found:", len(perhead_keys))
print(perhead_keys)

fact_perhead_parts = []

for k in perhead_keys:
    df = datasets[k]
    long_df = wide_to_long_finance(df=df, dataset_key=k, metric_type="PerHead")
    fact_perhead_parts.append(long_df)

fact_perhead = pd.concat(fact_perhead_parts, ignore_index=True)

validate_fact_table(fact_perhead, "Fact_PerHead")
fact_perhead.head()

# Build Fact_PctTotal

pcttotal_keys = [k for k in datasets.keys() if k.endswith("expenditure per cent of total")]
print("PctTotal datasets found:", len(pcttotal_keys))
print(pcttotal_keys)

fact_pcttotal_parts = []

for k in pcttotal_keys:
    df = datasets[k]
    long_df = wide_to_long_finance(df=df, dataset_key=k, metric_type="PctTotal")
    fact_pcttotal_parts.append(long_df)

fact_pcttotal = pd.concat(fact_pcttotal_parts, ignore_index=True)

validate_fact_table(fact_pcttotal, "Fact_PctTotal")
fact_pcttotal.head()

# Do all three fact tables cover the same boards?

print("Boards in Expenditure:", sorted(fact_expenditure["HealthBoard"].unique()))
print("Boards in PerHead:", sorted(fact_perhead["HealthBoard"].unique()))
print("Boards in PctTotal:", sorted(fact_pcttotal["HealthBoard"].unique()))


# Do all three cover the same FinancialYears?

print("FYs in Expenditure:", len(fact_expenditure["FinancialYear"].unique()))
print("FYs in PerHead:", len(fact_perhead["FinancialYear"].unique()))
print("FYs in PctTotal:", len(fact_pcttotal["FinancialYear"].unique()))


# Any unexpected missing FinancialYear extraction?

print("Missing FY in Expenditure:", fact_expenditure["FinancialYear"].isna().sum())
print("Missing FY in PerHead:", fact_perhead["FinancialYear"].isna().sum())
print("Missing FY in PctTotal:", fact_pcttotal["FinancialYear"].isna().sum())



# Diagnose duplicates in Fact_PerHead and Fact_PctTotal


# Diagnose the actual category column name in each fact table
print("Fact_Expenditure columns:", list(fact_expenditure.columns))
print("Fact_PerHead columns:", list(fact_perhead.columns))
print("Fact_PctTotal columns:", list(fact_pcttotal.columns))



### DEBUG

# STEP 1 — RAW duplicate check (before melt) for PerHead and PctTotal datasets

CATEGORY_COL = "Category breakdown"

perhead_keys = [k for k in datasets.keys() if k.endswith("expenditure per head")]
pcttotal_keys = [k for k in datasets.keys() if k.endswith("expenditure per cent of total")]

def raw_dup_summary(df, dataset_name):
    if CATEGORY_COL not in df.columns:
        print(f"\n{dataset_name}: MISSING '{CATEGORY_COL}' column. Columns are: {list(df.columns)}")
        return

    # Strip to prevent whitespace-based false uniqueness
    s = df[CATEGORY_COL].astype(str).str.strip()

    row_count = len(df)
    uniq_count = s.nunique(dropna=False)

    dup_counts = s.value_counts()
    dup_labels = dup_counts[dup_counts > 1]

    print(f"\n{dataset_name}")
    print(f"Rows: {row_count} | Unique categories: {uniq_count} | Duplicate category labels: {len(dup_labels)}")

    if len(dup_labels) > 0:
        print("Top duplicated categories (label -> count):")
        print(dup_labels.head(10))

print("==== RAW DUP CHECK: PER HEAD FILES ====")
for k in perhead_keys:
    raw_dup_summary(datasets[k], k)

print("\n==== RAW DUP CHECK: % OF TOTAL FILES ====")
for k in pcttotal_keys:
    raw_dup_summary(datasets[k], k)


## Create a unified “Fact_Finance” table 

CATEGORY_COL = "Category breakdown"
GRAIN_COLS = ["HealthBoard", "FinancialYear", CATEGORY_COL, "MetricType"]

# ---------------------------
# Step 2.3.1 — Standardise schemas (columns + types) before union
# ---------------------------

def standardise_fact_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required_cols = ["HealthBoard", "FinancialYear", CATEGORY_COL, "MetricType", "Value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Found: {list(df.columns)}")

    out = df[required_cols].copy()

    # Clean text fields
    out["HealthBoard"] = out["HealthBoard"].astype(str).str.strip()
    out["FinancialYear"] = out["FinancialYear"].astype(str).str.strip()
    out[CATEGORY_COL] = out[CATEGORY_COL].astype(str).str.strip()
    out["MetricType"] = out["MetricType"].astype(str).str.strip()

    # Numeric
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")

    return out

fact_expenditure_std = standardise_fact_schema(fact_expenditure, "fact_expenditure")
fact_perhead_std = standardise_fact_schema(fact_perhead, "fact_perhead")
fact_pcttotal_std = standardise_fact_schema(fact_pcttotal, "fact_pcttotal")

print("Schemas standardised:")
print("Expenditure:", fact_expenditure_std.shape)
print("PerHead:", fact_perhead_std.shape)
print("PctTotal:", fact_pcttotal_std.shape)





# ---------------------------
# Step 2.3.2 — Union (append) into Fact_Finance
# ---------------------------

fact_finance = pd.concat(
    [fact_expenditure_std, fact_perhead_std, fact_pcttotal_std],
    ignore_index=True
)

print("\nFact_Finance created. Shape:", fact_finance.shape)
print("MetricType counts:")
print(fact_finance["MetricType"].value_counts(dropna=False))



# ---------------------------
# Step 2.3.3 — Integrity checks (must pass before Power BI)
# ---------------------------

# Check A: Nulls
print("\nNull check:")
print(fact_finance.isna().sum())


# Check B: Duplicate grain rows
dup_count = fact_finance.duplicated(subset=GRAIN_COLS).sum()
print("\nDuplicate grain rows in Fact_Finance:", dup_count)




# Check C: MetricType only contains expected values
expected_metrics = {"Expenditure", "PerHead", "PctTotal"}
found_metrics = set(fact_finance["MetricType"].unique())
unexpected = sorted(found_metrics - expected_metrics)

print("\nMetricType values found:", sorted(found_metrics))
if unexpected:
    print("WARNING: Unexpected MetricType values:", unexpected)
    
    


coverage_summary = (
    fact_finance.groupby(["HealthBoard", "MetricType"])
               .size()
               .reset_index(name="rows")
               .sort_values(["HealthBoard", "MetricType"])
)

print("\nCoverage summary (rows by HealthBoard and MetricType):")
print(coverage_summary)




fact_finance.to_csv("fact_finance.csv", index=False)
print("\nSaved: fact_finance.csv")





# ============================================================
# PHASE 3.1 — Structural Validation (CODE)
# Focus:
# 3.1.1 Category completeness by board
# 3.1.2 Year completeness by board
# 3.1.3 Grain uniqueness
#
# Assumes you already created: fact_finance
# Columns expected:
#   HealthBoard, FinancialYear, Category breakdown, MetricType, Value
# ============================================================

import pandas as pd
import numpy as np

CATEGORY_COL = "Category breakdown"
GRAIN_COLS = ["HealthBoard", "FinancialYear", CATEGORY_COL, "MetricType"]

# -----------------------------
# Step 3.1.0 — Sanity check
# -----------------------------
required_cols = ["HealthBoard", "FinancialYear", CATEGORY_COL, "MetricType", "Value"]
missing = [c for c in required_cols if c not in fact_finance.columns]
if missing:
    raise ValueError(f"fact_finance is missing columns: {missing}. Found: {list(fact_finance.columns)}")
    
    

# Keep a clean copy for validation work (no surprises)
ff = fact_finance[required_cols].copy()
ff["HealthBoard"] = ff["HealthBoard"].astype(str).str.strip()
ff["FinancialYear"] = ff["FinancialYear"].astype(str).str.strip()
ff[CATEGORY_COL] = ff[CATEGORY_COL].astype(str).str.strip()
ff["MetricType"] = ff["MetricType"].astype(str).str.strip()

print("fact_finance shape:", ff.shape)
print("Boards:", sorted(ff["HealthBoard"].unique()))
print("MetricTypes:", sorted(ff["MetricType"].unique()))
print("FY range preview:", sorted(ff["FinancialYear"].unique())[:5], "...", sorted(ff["FinancialYear"].unique())[-5:])

# ============================================================
# 3.1.1 Category completeness by board
# ============================================================

# Step 3.1.1a — Build the "expected" category set per MetricType
# (We do it per MetricType because PerHead has known gaps.)
expected_categories_by_metric = {
    mt: set(ff.loc[ff["MetricType"] == mt, CATEGORY_COL].unique())
    for mt in ff["MetricType"].unique()
}

# Step 3.1.1b — For each board + metric, check missing categories vs expected set
cat_completeness_rows = []

for mt, expected_set in expected_categories_by_metric.items():
    for hb in sorted(ff["HealthBoard"].unique()):
        observed_set = set(ff.loc[(ff["MetricType"] == mt) & (ff["HealthBoard"] == hb), CATEGORY_COL].unique())
        missing_cats = sorted(expected_set - observed_set)
        extra_cats = sorted(observed_set - expected_set)  # usually empty by definition, but kept for debugging

        cat_completeness_rows.append({
            "MetricType": mt,
            "HealthBoard": hb,
            "ExpectedCategoryCount": len(expected_set),
            "ObservedCategoryCount": len(observed_set),
            "MissingCategoryCount": len(missing_cats),
            "ExtraCategoryCount": len(extra_cats),
            "MissingCategoriesPreview": ", ".join(missing_cats[:10])  # preview only
        })

cat_completeness = pd.DataFrame(cat_completeness_rows)

print("\n=== 3.1.1 Category completeness by board (summary) ===")
print(cat_completeness.sort_values(["MetricType", "MissingCategoryCount"], ascending=[True, False]).head(20))

# Optional: show only failures
cat_failures = cat_completeness[cat_completeness["MissingCategoryCount"] > 0].copy()
print("\nCategory completeness failures count:", len(cat_failures))
if len(cat_failures) > 0:
    print(cat_failures.sort_values(["MetricType", "MissingCategoryCount"], ascending=[True, False]).head(30))


# ============================================================
# 3.1.2 Year completeness by board
# ============================================================

# Step 3.1.2a — Build expected year set per MetricType
expected_years_by_metric = {
    mt: set(ff.loc[ff["MetricType"] == mt, "FinancialYear"].unique())
    for mt in ff["MetricType"].unique()
}

# Step 3.1.2b — For each board + metric, check missing years
year_completeness_rows = []

for mt, expected_years in expected_years_by_metric.items():
    for hb in sorted(ff["HealthBoard"].unique()):
        observed_years = set(ff.loc[(ff["MetricType"] == mt) & (ff["HealthBoard"] == hb), "FinancialYear"].unique())
        missing_years = sorted(expected_years - observed_years)

        year_completeness_rows.append({
            "MetricType": mt,
            "HealthBoard": hb,
            "ExpectedYearCount": len(expected_years),
            "ObservedYearCount": len(observed_years),
            "MissingYearCount": len(missing_years),
            "MissingYearsPreview": ", ".join(missing_years[:10])
        })

year_completeness = pd.DataFrame(year_completeness_rows)

print("\n=== 3.1.2 Year completeness by board (summary) ===")
print(year_completeness.sort_values(["MetricType", "MissingYearCount"], ascending=[True, False]).head(20))

# Optional: show only failures
year_failures = year_completeness[year_completeness["MissingYearCount"] > 0].copy()
print("\nYear completeness failures count:", len(year_failures))
if len(year_failures) > 0:
    print(year_failures.sort_values(["MetricType", "MissingYearCount"], ascending=[True, False]).head(30))


# ============================================================
# 3.1.3 Grain uniqueness (duplicate keys at target grain)
# ============================================================

dup_mask = ff.duplicated(subset=GRAIN_COLS, keep=False)
dup_count = dup_mask.sum()

print("\n=== 3.1.3 Grain uniqueness ===")
print("Duplicate grain rows:", dup_count)

if dup_count > 0:
    dup_rows = ff.loc[dup_mask].sort_values(GRAIN_COLS).copy()
    print("\nPreview of duplicate grain rows (first 25):")
    print(dup_rows.head(25))

    # Optional: show how many duplicates per grain key
    dup_key_summary = (
        dup_rows.groupby(GRAIN_COLS)
                .size()
                .reset_index(name="RowCount")
                .sort_values("RowCount", ascending=False)
    )
    print("\nDuplicate grain key summary (top 20):")
    print(dup_key_summary.head(20))


# ============================================================
# (Optional) Build a simple "Validation Report" table for 3.1
# ============================================================

validation_report = []

# Category completeness pass/fail — pass if no failures for Expenditure and PctTotal
# (PerHead may legitimately have gaps; we still report it but don't fail the whole dataset by default.)
def pass_for_metrics(df_failures: pd.DataFrame, metrics_to_enforce: list) -> bool:
    return df_failures[df_failures["MetricType"].isin(metrics_to_enforce)].empty

cat_pass = pass_for_metrics(cat_failures, ["Expenditure", "PctTotal"])
year_pass = pass_for_metrics(year_failures, ["Expenditure", "PctTotal"])
grain_pass = (dup_count == 0)

validation_report.append({
    "CheckName": "3.1.1 Category completeness (Expenditure + PctTotal enforced)",
    "Status": "PASS" if cat_pass else "FAIL",
    "Notes": "PerHead gaps are tracked separately; enforcement applied to Expenditure and PctTotal."
})

validation_report.append({
    "CheckName": "3.1.2 Year completeness (Expenditure + PctTotal enforced)",
    "Status": "PASS" if year_pass else "FAIL",
    "Notes": "PerHead gaps are tracked separately; enforcement applied to Expenditure and PctTotal."
})

validation_report.append({
    "CheckName": "3.1.3 Grain uniqueness (no duplicate keys)",
    "Status": "PASS" if grain_pass else "FAIL",
    "Notes": "Uniqueness at HealthBoard+FinancialYear+Category breakdown+MetricType."
})

validation_report_df = pd.DataFrame(validation_report)
print("\n=== Validation Report (Phase 3.1) ===")
print(validation_report_df)

# If you want to export these:
# cat_completeness.to_csv("validation_3_1_1_category_completeness.csv", index=False)
# year_completeness.to_csv("validation_3_1_2_year_completeness.csv", index=False)
# validation_report_df.to_csv("validation_3_1_report.csv", index=False)





























































