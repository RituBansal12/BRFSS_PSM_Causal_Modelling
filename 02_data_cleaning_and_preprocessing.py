# Rename this script to data_cleaning_and_preprocessing.py for pipeline continuity
# All cleaning and preprocessing steps are performed here
# Output is cleaned_llcp2023.csv for downstream scripts

import pandas as pd
import numpy as np
import logging
import os
import warnings

warnings.filterwarnings("ignore")

def load_metadata(metadata_path):
    """Load metadata and build mappings and rules."""
    meta = pd.read_csv(metadata_path)
    # Build mappings: {var: {value: label}}
    mappings = {}
    for var in meta['SAS Variable Name'].unique():
        sub = meta[meta['SAS Variable Name'] == var]
        mappings[var] = dict(zip(sub['Value'].astype(str), sub['Value Label']))
    # Build notes dict
    notes = meta.groupby('SAS Variable Name')['Notes'].first().to_dict() if 'Notes' in meta else {}
    return mappings, notes

def load_raw_data(data_path):
    return pd.read_csv(data_path)

# Special codes to NaN
SPECIAL_CODES = set(['7','77','777','9','99','999','7777','9999','777777','999999','7777777','9999999','BLANK',''])

# Variable-specific cleaning logic
# Example for categorical

def clean_categorical(df, mappings, var):
    if var not in df.columns:
        return df
    df[var] = df[var].astype(str)
    df[var] = df[var].replace(list(SPECIAL_CODES), np.nan)
    if var in mappings:
        df[var + '_label'] = df[var].map(mappings[var])
    return df

def clean_boolean(df, var):
    if var not in df.columns:
        return df
    df[var] = df[var].replace(list(SPECIAL_CODES), np.nan)
    df[var] = df[var].map({1: True, 2: False})
    return df

def clean_numerical(df, var, notes=None):
    if var not in df.columns:
        return df
    df[var] = pd.to_numeric(df[var], errors='coerce')
    # Example: handle 88/888 as zero if applicable
    if notes and 'None' in str(notes):
        df[var] = df[var].replace({88: 0, 888: 0})
    df[var] = df[var].replace(list(map(int, filter(lambda x: x.isdigit(), SPECIAL_CODES))), np.nan)
    return df

def clean_dates(df, year_var, month_var, day_var, new_var):
    if all(v in df.columns for v in [year_var, month_var, day_var]):
        df[new_var] = pd.to_datetime(dict(year=df[year_var], month=df[month_var], day=df[day_var]), errors='coerce')
    return df

def main():
    logging.basicConfig(filename='clean_llcp2023.log', level=logging.INFO)
    meta_path = 'data/LLCP2023_metadata.csv'
    data_path = 'data/LLCP2023.csv'
    mappings, notes = load_metadata(meta_path)
    df = load_raw_data(data_path)

    # Clean categorical variables
    categorical_vars = [v for v in mappings.keys() if v in df.columns]
    for var in categorical_vars:
        df = clean_categorical(df, mappings, var)
        logging.info(f'Cleaned categorical: {var}')
    logging.info(f'Rows after categorical cleaning: {len(df)}')

    # Clean boolean variables
    boolean_vars = ['EXERANY2','BPMEDS1','SMOKE100','COVIDPO1']
    for var in boolean_vars:
        df = clean_boolean(df, var)
        logging.info(f'Cleaned boolean: {var}')
    logging.info(f'Rows after boolean cleaning: {len(df)}')

    # Clean numerical variables
    numerical_vars = ['PHYSHLTH','MENTHLTH','POORHLTH','DIABAGE4','JOINPAI2','LCSFIRST','LCSLAST','LCSNUMCG','MARIJAN1']
    for var in numerical_vars:
        df = clean_numerical(df, var, notes.get(var))
        logging.info(f'Cleaned numerical: {var}')
    logging.info(f'Rows after numerical cleaning: {len(df)}')

    # Clean date variables
    df = clean_dates(df, 'IYEAR', 'IMONTH', 'IDAY', 'INTERVIEW_DATE')
    logging.info('Constructed INTERVIEW_DATE')
    logging.info(f'Rows after date cleaning: {len(df)}')

    # Recode _MENT14D to binary
    if '_MENT14D' in df.columns:
        def recode_mental(x):
            try:
                val = float(x)
                if val == 1.0:
                    return 0
                elif val in [2.0, 3.0]:
                    return 1
                else:
                    return np.nan
            except:
                return np.nan
        df['_MENTAL_HEALTH_BINARY'] = df['_MENT14D'].apply(recode_mental)
        logging.info(f'Rows with valid _MENTAL_HEALTH_BINARY: {df["_MENTAL_HEALTH_BINARY"].notna().sum()}')

    # Recode _TOTINDA to binary
    if '_TOTINDA' in df.columns:
        def recode_exercise(x):
            try:
                val = float(x)
                if val == 1.0:
                    return 1
                elif val == 2.0:
                    return 0
                else:
                    return np.nan
            except:
                return np.nan
        df['_EXERCISE_BINARY'] = df['_TOTINDA'].apply(recode_exercise)
        logging.info(f'Rows with valid _EXERCISE_BINARY: {df["_EXERCISE_BINARY"].notna().sum()}')

    # Only keep rows where both binary columns are present and valid
    if '_MENTAL_HEALTH_BINARY' in df.columns and '_EXERCISE_BINARY' in df.columns:
        df = df[df['_MENTAL_HEALTH_BINARY'].notna() & df['_EXERCISE_BINARY'].notna()]
        df['_MENTAL_HEALTH_BINARY'] = df['_MENTAL_HEALTH_BINARY'].astype(int)
        df['_EXERCISE_BINARY'] = df['_EXERCISE_BINARY'].astype(int)
        logging.info(f'Rows after filtering for both binaries: {len(df)}')

    # Save cleaned data as CSV for continuity
    df.to_csv('data/cleaned_llcp2023.csv', index=False)
    logging.info('Saved cleaned data to data/cleaned_llcp2023.csv')

if __name__ == '__main__':
    main()
