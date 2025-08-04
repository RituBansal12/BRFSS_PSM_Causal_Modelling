 #!/usr/bin/env python3
"""
Data Preprocessing Script for BRFSS 2023 Dataset
Converts raw BRFSS data into a processed dataset with engineered features
for counterfactual modeling analysis.
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_filter_metadata(metadata_path="data/LLCP2023_metadata.csv"):
    """Load metadata and filter to relevant sections for analysis."""
    print("Loading metadata...")
    metadata = pd.read_csv(metadata_path)
    
    # Clean section names
    def clean_section_name(name):
        if pd.isnull(name):
            return name
        name = name.lower()
        name = re.sub(r'\(.*?\)', '', name)
        name = re.sub(r'[^a-z0-9 ]+', '', name)
        name = re.sub(r'\s+', '_', name.strip())
        return name
    
    metadata['Section Name Clean'] = metadata['Section Name'].apply(clean_section_name)
    
    # Define sections of interest
    sections_interest = [
        'adversechildhoodexperiences', 'alcoholconsumption', 'chronichealthconditions',
        'demographics', 'disability', 'healthcareaccess', 'healthydays',
        'marijuanause', 'othertobaccouse', 'respondentsex', 'sexualorientation',
        'socialdeterminants', 'tobaccouse', 'urbanrural', 'exercise'
    ]
    
    # Filter metadata to relevant sections
    df_subset = metadata[metadata['Section Name Clean'].isin(sections_interest)]
    df_subset = df_subset[['SAS Variable Name', 'Section Name Clean', 'Label']].drop_duplicates()
    
    return df_subset

def get_target_variables():
    """Define the target variables for analysis."""
    variables = [
        "SEXVAR", "PHYSHLTH", "MENTHLTH", "POORHLTH", "PRIMINS1", "PERSDOC3",
        "MEDCOST1", "CHECKUP1", "EXERANY2", "EXRACT12", "EXEROFT1", "EXERHMM1",
        "EXRACT22", "EXEROFT2", "EXERHMM2", "STRENGTH", "ADDEPEV3", "MARITAL",
        "EDUCA", "RENTHOM1", "VETERAN3", "EMPLOY1", "CHILDREN", "INCOME3",
        "PREGNANT", "WEIGHT2", "HEIGHT3", "DEAF", "BLIND", "DECIDE", "DIFFWALK",
        "DIFFDRES", "DIFFALON", "SMOKE100", "SMOKDAY2", "ECIGNOW2", "ALCDAY4",
        "AVEDRNK3", "DRNK3GE5", "MAXDRNKS", "SOMALE", "SOFEMALE", "MARIJAN1",
        "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC", "ACEPUNCH",
        "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX", "ACEADSAF",
        "ACEADNED", "LSATISFY", "EMTSUPRT", "SDLONELY", "SDHEMPLY", "FOODSTMP",
        "SDHFOOD1", "SDHBILLS", "SDHUTILS", "SDHTRNSP", "SDHSTRE1", "_METSTAT", "_URBSTAT"
    ]
    return variables

def process_ace_variables(data):
    """Process Adverse Childhood Experiences (ACE) variables."""
    print("Processing ACE variables...")
    
    ace_value_mapping = {
        'ACEDEPRS': {1: 1, 2: 0, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEDRINK': {1: 1, 2: 0, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEDRUGS': {1: 1, 2: 0, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEPRISN': {1: 1, 2: 0, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEDIVRC': {1: 1, 2: 0, 7: None, 8: None, 9: None, 'BLANK': None, None: None},
        'ACEPUNCH': {1: 0, 2: 1, 3: 1, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEHURT1': {1: 0, 2: 1, 3: 1, 7: None, 9: None, 'BLANK': None, None: None},
        'ACESWEAR': {1: 0, 2: 1, 3: 1, 7: None, 9: None, 'BLANK': None, None: None},
        'ACETOUCH': {1: 0, 2: 1, 3: 1, 7: None, 9: None, 'BLANK': None, None: None},
        'ACETTHEM': {1: 0, 2: 1, 3: 1, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEHVSEX': {1: 0, 2: 1, 3: 1, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEADSAF': {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 7: None, 9: None, 'BLANK': None, None: None},
        'ACEADNED': {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 7: None, 9: None, 'BLANK': None, None: None}
    }
    
    ace_columns = list(ace_value_mapping.keys())
    
    # Apply mapping to each column
    for col in ace_columns:
        if col in data.columns:
            data[col] = data[col].map(ace_value_mapping[col])
    
    # Convert None to np.nan
    data[ace_columns] = data[ace_columns].applymap(lambda x: np.nan if x is None else x)
    
    # Calculate total ACE score
    data['ACE_TOTAL'] = data[ace_columns].sum(axis=1, skipna=True)
    
    # Normalize ACE score
    scaler = MinMaxScaler((0, 1))
    data['ACE_NORMALIZED'] = scaler.fit_transform(data[['ACE_TOTAL']])
    
    return data

def process_alcohol_variables(data):
    """Process alcohol consumption variables."""
    print("Processing alcohol variables...")
    
    def map_alcohol_severity_weighted(row):
        # Frequency: ALCDAY4
        alcfreq = row['ALCDAY4']
        if 101 <= alcfreq <= 199:
            freq_score = alcfreq - 100
        elif 201 <= alcfreq <= 299:
            freq_score = (alcfreq - 200) / 4.3
        else:
            freq_score = np.nan
        
        # Normalize to 0–1
        norm_freq = freq_score / 30 if pd.notna(freq_score) else np.nan
        
        # Drinks per day: AVEDRNK3
        avg_drinks = row['AVEDRNK3']
        norm_drinks = avg_drinks / 15 if 1 <= avg_drinks <= 76 else np.nan
        
        # Binge days: DRNK3GE5
        binge = row['DRNK3GE5']
        norm_binge = binge / 30 if 1 <= binge <= 76 else np.nan
        
        # Max drinks in a session: MAXDRNKS
        maxdrinks = row['MAXDRNKS']
        norm_max = maxdrinks / 15 if 1 <= maxdrinks <= 76 else np.nan
        
        # Combine with equal weights
        components = [norm_freq, norm_drinks, norm_binge, norm_max]
        valid_components = [x for x in components if pd.notna(x)]
        
        if valid_components:
            weighted_score = np.mean(valid_components)
        else:
            weighted_score = np.nan
        
        return weighted_score
    
    data['ALCOHOL_SEVERITY'] = data.apply(map_alcohol_severity_weighted, axis=1)
    
    # Normalize alcohol severity
    scaler = MinMaxScaler((0, 1))
    data['ALCOHOL_SEVERITY_NORMALIZED'] = scaler.fit_transform(data[['ALCOHOL_SEVERITY']])
    
    return data

def process_demographic_variables(data):
    """Process demographic variables."""
    print("Processing demographic variables...")
    
    # Gender mapping
    sexvar_mapping = {1: 'Male', 2: 'Female'}
    data['SEXVAR'] = data['SEXVAR'].map(sexvar_mapping)
    
    # Other demographic mappings
    marital_mapping = {
        1: "Married", 2: "Divorced", 3: "Widowed", 4: "Separated",
        5: "Never married", 6: "A member of an unmarried couple",
        9: None, "": None
    }
    
    educa_mapping = {
        1: "Never attended school or only kindergarten",
        2: "Elementary", 3: "Some high school", 4: "High school graduate",
        5: "Some college/tech school", 6: "College graduate",
        9: None, "": None
    }
    
    renthom1_mapping = {1: "Own", 2: "Rent", 3: "Other", 7: None, 9: None, "": None}
    veteran3_mapping = {1: "Yes", 2: "No", 7: None, 9: None, "": None}
    
    employ1_mapping = {
        1: "Employed for wages", 2: "Self-employed", 3: "Out of work ≥1 year",
        4: "Out of work <1 year", 5: "Homemaker", 6: "Student",
        7: "Retired", 8: "Unable to work", 9: None, "": None
    }
    
    children_mapping = {**{i: str(i) for i in range(1, 88)}, 88: "", 99: None, "": None}
    
    income3_mapping = {
        1: "<$10k", 2: "$10k–15k", 3: "$15k–20k", 4: "$20k–25k",
        5: "$25k–35k", 6: "$35k–50k", 7: "$50k–75k", 8: "$75k–100k",
        9: "$100k–150k", 10: "$150k–200k", 11: "$200k+", 77: None,
        99: None, "": None
    }
    
    pregnant_mapping = {1: "Yes", 2: "No", 7: None, 9: None, "": None}
    
    def map_weight_height(value, special_codes=[7777, 9999]):
        if pd.isna(value) or value in special_codes:
            return np.nan
        else:
            return value
    
    # Apply mappings
    data['MARITAL'] = data['MARITAL'].map(marital_mapping)
    data['EDUCA'] = data['EDUCA'].map(educa_mapping)
    data['RENTHOM1'] = data['RENTHOM1'].map(renthom1_mapping)
    data['VETERAN3'] = data['VETERAN3'].map(veteran3_mapping)
    data['EMPLOY1'] = data['EMPLOY1'].map(employ1_mapping)
    data['CHILDREN'] = data['CHILDREN'].map(children_mapping)
    data['INCOME3'] = data['INCOME3'].map(income3_mapping)
    data['PREGNANT'] = data['PREGNANT'].map(pregnant_mapping)
    data['WEIGHT2'] = data['WEIGHT2'].apply(lambda x: map_weight_height(x))
    data['HEIGHT3'] = data['HEIGHT3'].apply(lambda x: map_weight_height(x))
    
    return data

def process_disability_variables(data):
    """Process disability variables."""
    print("Processing disability variables...")
    
    disability_mapping = {1: "Yes", 2: "No", 7: None, 9: None, "": None}
    disability_vars = ['DEAF', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON']
    
    # Create binary indicator columns
    for col in disability_vars:
        if col in data.columns:
            data[f'{col}_bin'] = data[col].apply(lambda x: 1 if x == 1 else 0)
    
    # Create disability index
    bin_cols = [f'{col}_bin' for col in disability_vars if f'{col}_bin' in data.columns]
    data['DISABILITY_INDEX'] = data[bin_cols].sum(axis=1)
    data['DISABILITY_INDEX_NORMALIZED'] = data['DISABILITY_INDEX'] / len(disability_vars)
    
    return data

def process_geographic_variables(data):
    """Process geographic variables."""
    print("Processing geographic variables...")
    
    metstat_mapping = {1: "Metropolitan", 2: "Nonmetropolitan", "": None}
    urbstat_mapping = {1: "Urban", 2: "Rural", "": None}
    
    data['METRO_AREA'] = data['_METSTAT'].map(metstat_mapping)
    data['URBAN_RURAL_AREA'] = data['_URBSTAT'].map(urbstat_mapping)
    
    return data

def process_health_variables(data):
    """Process health-related variables."""
    print("Processing health variables...")
    
    addepev3_mapping = {1: "Yes", 2: "No", 7: None, 9: None, np.nan: None}
    data['ADDEPEV3'] = data['ADDEPEV3'].map(addepev3_mapping)
    
    return data

def process_substance_use_variables(data):
    """Process substance use variables."""
    print("Processing substance use variables...")
    
    # Smoking index
    data['SMOKE100'] = data['SMOKE100'].map({1: 1, 2: 0, 7: 0, 9: 0})
    data['SMOKDAY2'] = data['SMOKDAY2'].map({1: 1, 2: 0.5, 3: 0, 7: 0, 9: 0})
    data['ECIGNOW2'] = data['ECIGNOW2'].map({2: 1, 3: 0.5, 4: 0, 1: 0, 7: 0, 9: 0})
    
    data['SMOKING_INDEX'] = data[['SMOKE100', 'SMOKDAY2', 'ECIGNOW2']].mean(axis=1)
    
    # Normalize smoking index
    smoking_min = data['SMOKING_INDEX'].min()
    smoking_max = data['SMOKING_INDEX'].max()
    data['SMOKING_INDEX_NORMALIZED'] = (data['SMOKING_INDEX'] - smoking_min) / (smoking_max - smoking_min)
    
    # Marijuana index
    data['MARIJAN1'] = np.where(
        data['MARIJAN1'].isin(range(1, 31)),
        data['MARIJAN1'],
        0
    )
    data['MARIJUANA_INDEX_NORMALIZED'] = data['MARIJAN1'] / 30
    
    return data

def process_sexual_orientation(data):
    """Process sexual orientation variables."""
    print("Processing sexual orientation variables...")
    
    def map_orientation(somale, sofemale):
        val = somale if pd.notna(somale) else sofemale
        
        if val == 1:
            return 1  # Gay or Lesbian
        elif val == 3:
            return 0.5  # Bisexual
        elif val in [2, 4, 7, 9] or pd.isna(val):
            return 0  # Straight or Other/Refused/Missing
        else:
            return 0
    
    data['SEXUAL_ORIENTATION_SCORE'] = data.apply(
        lambda row: map_orientation(row['SOMALE'], row['SOFEMALE']), axis=1
    )
    
    return data

def process_social_determinants(data):
    """Process social determinants variables."""
    print("Processing social determinants variables...")
    
    # Mapping dictionaries
    satisfaction_map = {1: 0, 2: 0.33, 3: 0.67, 4: 1, 7: np.nan, 9: np.nan, '': np.nan, np.nan: np.nan}
    emotional_support_map = {1: 0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1, 7: np.nan, 9: np.nan, '': np.nan, np.nan: np.nan}
    lonely_map = {1: 1, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0, 7: np.nan, 9: np.nan, '': np.nan, np.nan: np.nan}
    employment_loss_map = {1: 1, 2: 0, 7: np.nan, 9: np.nan, '': np.nan, np.nan: np.nan}
    food_stamps_map = {1: 1, 2: 0, 7: np.nan, 9: np.nan, '': np.nan, np.nan: np.nan}
    food_insecurity_map = {1: 1, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0, 7: np.nan, 9: np.nan, '': np.nan, np.nan: np.nan}
    bills_map = {1: 1, 2: 0, 7: np.nan, 9: np.nan, '': np.nan, np.nan: np.nan}
    
    mapping_dict = {
        'LSATISFY': satisfaction_map,
        'EMTSUPRT': emotional_support_map,
        'SDLONELY': lonely_map,
        'SDHEMPLY': employment_loss_map,
        'FOODSTMP': food_stamps_map,
        'SDHFOOD1': food_insecurity_map,
        'SDHBILLS': bills_map,
        'SDHUTILS': bills_map,
        'SDHTRNSP': bills_map,
        'SDHSTRE1': food_insecurity_map
    }
    
    indexes = {
        'financial_insecurity': ['SDHBILLS', 'SDHUTILS', 'SDHTRNSP'],
        'food_insecurity': ['FOODSTMP', 'SDHFOOD1'],
        'emotional_distress': ['LSATISFY', 'EMTSUPRT', 'SDLONELY', 'SDHSTRE1'],
        'facing_unemployment': ['SDHEMPLY']
    }
    
    weights = {
        'financial_insecurity': {'SDHBILLS': 0.4, 'SDHUTILS': 0.3, 'SDHTRNSP': 0.3},
        'food_insecurity': {'FOODSTMP': 0.5, 'SDHFOOD1': 0.5},
        'emotional_distress': {'LSATISFY': 0.25, 'EMTSUPRT': 0.25, 'SDLONELY': 0.25, 'SDHSTRE1': 0.25},
        'facing_unemployment': {'SDHEMPLY': 1.0}
    }
    
    # Pre-map all columns
    mapped_data = {}
    for col, mapping in mapping_dict.items():
        if col in data.columns:
            mapped_data[col] = data[col].map(mapping)
    
    # Calculate indexes using vectorized operations
    for idx_name, cols in indexes.items():
        available_cols = [col for col in cols if col in mapped_data]
        
        if available_cols:
            temp_df = pd.DataFrame({col: mapped_data[col] for col in available_cols})
            col_weights = {col: weights[idx_name][col] for col in available_cols}
            
            weighted_sums = pd.Series(0.0, index=data.index)
            weight_sums = pd.Series(0.0, index=data.index)
            
            for col in available_cols:
                weight = col_weights[col]
                mask = temp_df[col].notna()
                weighted_sums[mask] += temp_df[col][mask] * weight
                weight_sums[mask] += weight
            
            data[idx_name.upper() + '_INDEX'] = np.where(
                weight_sums > 0, 
                weighted_sums / weight_sums, 
                np.nan
            )
            
            # Normalize to 0-1 scale
            min_val = data[idx_name.upper() + '_INDEX'].min()
            max_val = data[idx_name.upper() + '_INDEX'].max()
            
            if max_val - min_val == 0:
                data[idx_name.upper() + '_INDEX_NORMALIZED'] = 0
            else:
                data[idx_name.upper() + '_INDEX_NORMALIZED'] = (
                    data[idx_name.upper() + '_INDEX'] - min_val) / (max_val - min_val)
        else:
            data[idx_name.upper() + '_INDEX'] = np.nan
            data[idx_name.upper() + '_INDEX_NORMALIZED'] = np.nan
    
    return data

def process_healthcare_access(data):
    """Process healthcare access variables."""
    print("Processing healthcare access variables...")
    
    primins1_map = {
        1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4, 6: 0.4, 7: 0.6, 8: 0.5,
        9: 0.5, 10: 0.4, 88: 0.0, 77: np.nan, 99: np.nan, np.nan: np.nan
    }
    
    persdoc3_map = {1: 1.0, 2: 1.0, 3: 0.0, 7: np.nan, 9: np.nan, np.nan: np.nan}
    medcost1_map = {1: 0.0, 2: 1.0, 7: np.nan, 9: np.nan, np.nan: np.nan}
    checkup1_map = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.2, 7: np.nan, 8: 0.0, 9: np.nan, np.nan: np.nan}
    
    data['PRIMINS1'] = data['PRIMINS1'].map(primins1_map)
    data['PERSDOC3'] = data['PERSDOC3'].map(persdoc3_map)
    data['MEDCOST1'] = data['MEDCOST1'].map(medcost1_map)
    data['CHECKUP1'] = data['CHECKUP1'].map(checkup1_map)
    
    weights = {'PRIMINS1': 0.25, 'PERSDOC3': 0.25, 'MEDCOST1': 0.25, 'CHECKUP1': 0.25}
    mapped_cols = list(weights.keys())
    
    # Calculate weighted index
    temp_df = data[mapped_cols].copy()
    weighted_sums = pd.Series(0.0, index=data.index)
    weight_sums = pd.Series(0.0, index=data.index)
    
    for col in mapped_cols:
        weight = weights[col]
        mask = temp_df[col].notna()
        weighted_sums[mask] += temp_df[col][mask] * weight
        weight_sums[mask] += weight
    
    data['HEALTHCAREACCESS_INDEX_WEIGHTED'] = np.where(
        weight_sums > 0, 
        weighted_sums / weight_sums, 
        np.nan
    )
    
    # Normalize index
    min_val = data['HEALTHCAREACCESS_INDEX_WEIGHTED'].min()
    max_val = data['HEALTHCAREACCESS_INDEX_WEIGHTED'].max()
    data['HEALTHCAREACCESS_INDEX_NORMALIZED'] = (data['HEALTHCAREACCESS_INDEX_WEIGHTED'] - min_val) / (max_val - min_val)
    
    return data

def process_mental_health(data):
    """Process mental health variables."""
    print("Processing mental health variables...")
    
    def map_menthlth(val):
        if pd.isna(val):
            return np.nan
        
        try:
            val_float = float(val)
            if val_float in [77, 88, 99]:
                return np.nan
            if 1 <= val_float <= 30:
                return val_float / 30
            else:
                return np.nan
        except (ValueError, TypeError):
            return np.nan
    
    data['MENTHLTH_MAPPED'] = data['MENTHLTH'].apply(map_menthlth)
    
    return data

def process_exercise_variables(data):
    """Process exercise and physical activity variables."""
    print("Processing exercise variables...")
    
    def map_frequency(val, never_code=None):
        try:
            val = int(val)
            if 101 <= val <= 199:
                return val - 100
            elif 201 <= val <= 299:
                return (val - 200) / 4.345
            elif never_code is not None and val == never_code:
                return 0
            else:
                return np.nan
        except:
            return np.nan
    
    # Map strength exercise
    data['STRENGTH_WEEKLY'] = data['STRENGTH'].apply(lambda x: map_frequency(x, never_code=888))
    
    # Map aerobic exercise frequencies
    data['EXEROFT1_WEEKLY'] = data['EXEROFT1'].apply(map_frequency)
    data['EXEROFT2_WEEKLY'] = data['EXEROFT2'].apply(map_frequency)
    
    # Activity type mapping
    activity_map = {
        1: 'walking', 2: 'running_jogging', 3: 'gardening', 4: 'bicycling',
        5: 'aerobics_class', 6: 'calisthenics', 7: 'elliptical',
        8: 'household_activities', 9: 'weight_lifting', 10: 'yoga_pilates_tai_chi',
        11: 'other', 77: np.nan, 88: 'no_other_activity', 99: np.nan,
    }
    
    data['EXRACT12_ACTIVITY'] = data['EXRACT12'].map(activity_map).astype('category')
    data['EXRACT22_ACTIVITY'] = data['EXRACT22'].map(activity_map).astype('category')
    
    # Aerobic activity indicator
    aerobic_codes = {1, 2, 4, 5, 7}
    data['EXRACT12_AEROBIC'] = data['EXRACT12'].apply(lambda x: int(x) in aerobic_codes if pd.notna(x) else False)
    data['EXRACT22_AEROBIC'] = data['EXRACT22'].apply(lambda x: int(x) in aerobic_codes if pd.notna(x) else False)
    
    # Minutes/hours mapping
    def map_time_minutes(val):
        try:
            val = int(val)
            if 1 <= val <= 759:
                return val
            elif 800 <= val <= 959:
                return (val - 800) * 60
            else:
                return np.nan
        except:
            return np.nan
    
    data['EXERHMM1_MIN'] = data['EXERHMM1'].apply(map_time_minutes)
    data['EXERHMM2_MIN'] = data['EXERHMM2'].apply(map_time_minutes)
    
    return data

def clean_final_dataset(data):
    """Remove original variables that have been processed into new features."""
    print("Cleaning final dataset...")
    
    drop_columns = [
        'ACEDEPRS', 'ACEDRINK', 'ACEDRUGS', 'ACEPRISN', 'ACEDIVRC', 'ACEPUNCH',
        'ACEHURT1', 'ACESWEAR', 'ACETOUCH', 'ACETTHEM', 'ACEHVSEX', 'ACEADSAF', 'ACEADNED',
        'ALCDAY4', 'AVEDRNK3', 'DRNK3GE5', 'MAXDRNKS',
        'DEAF', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON',
        '_METSTAT', '_URBSTAT',
        'SMOKE100', 'SMOKDAY2', 'ECIGNOW2',
        'MARIJAN1',
        'SOMALE', 'SOFEMALE',
        'LSATISFY', 'EMTSUPRT', 'SDLONELY', 'SDHEMPLY', 'FOODSTMP',
        'SDHFOOD1', 'SDHBILLS', 'SDHUTILS', 'SDHTRNSP', 'SDHSTRE1',
        'PRIMINS1', 'PERSDOC3', 'MEDCOST1', 'CHECKUP1',
        'PHYSHLTH', 'MENTHLTH', 'POORHLTH',
        'EXERANY2', 'EXRACT12', 'EXEROFT1', 'EXERHMM1', 'EXRACT22', 'EXEROFT2', 'EXERHMM2', 'STRENGTH',
        'DEAF_label', 'BLIND_label', 'DECIDE_label', 'DIFFWALK_label', 'DIFFDRES_label', 'DIFFALON_label',
        'DEAF_bin', 'BLIND_bin', 'DECIDE_bin', 'DIFFWALK_bin', 'DIFFDRES_bin', 'DIFFALON_bin'
    ]
    
    # Only drop columns that exist
    existing_drop_columns = [col for col in drop_columns if col in data.columns]
    data = data.drop(columns=existing_drop_columns)
    
    return data

def main():
    """Main preprocessing pipeline."""
    print("Starting BRFSS 2023 data preprocessing...")
    
    # Load metadata and get target variables
    metadata_subset = load_and_filter_metadata()
    target_variables = get_target_variables()
    
    # Load main dataset
    print("Loading main dataset...")
    data = pd.read_csv("data/LLCP2023.csv")
    
    # Subset to target variables
    available_vars = [var for var in target_variables if var in data.columns]
    data = data[available_vars]
    
    print(f"Dataset shape: {data.shape}")
    print(f"Available variables: {len(available_vars)}")
    
    # Process all variable groups
    data = process_ace_variables(data)
    data = process_alcohol_variables(data)
    data = process_demographic_variables(data)
    data = process_disability_variables(data)
    data = process_geographic_variables(data)
    data = process_health_variables(data)
    data = process_substance_use_variables(data)
    data = process_sexual_orientation(data)
    data = process_social_determinants(data)
    data = process_healthcare_access(data)
    data = process_mental_health(data)
    data = process_exercise_variables(data)
    
    # Clean final dataset
    data = clean_final_dataset(data)
    
    # Save processed dataset
    output_path = "data/LLCP2023_processed.csv"
    data.to_csv(output_path, index=False)
    
    print(f"Preprocessing completed!")
    print(f"Final dataset shape: {data.shape}")
    print(f"Processed data saved to: {output_path}")
    
    return data

if __name__ == "__main__":
    processed_data = main() 