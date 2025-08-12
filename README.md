# PSM and Causal Modeling: Exercise Impact on Mental Health

This project analyzes the causal effect of exercise on mental health using the 2023 Behavioral Risk Factor Surveillance System (BRFSS) dataset. The analysis employs propensity score matching and causal inference methods to estimate the treatment effect of exercise participation on mental health outcomes.

## Article 
1. https://medium.com/@ritu.bansalrb00/causal-questions-in-a-correlational-world-going-beyond-rcts-ace0f0b72c6d

2. https://medium.com/@ritu.bansalrb00/causal-questions-in-a-correlation-world-part-ii-measuring-how-much-and-for-whom-076009895f14


## Project Overview

The analysis follows a systematic approach to causal inference:
1. **Data Extraction**: Convert BRFSS XPT format to CSV and extract metadata
2. **Data Preprocessing**: Clean and engineer features for analysis
3. **Propensity Score Matching**: Create balanced treatment/control groups
4. **Causal Model Fitting**: Estimate treatment effects using multiple methods

## File Structure

### Core Analysis Scripts

#### `01_data_extraction.py`
- **Purpose**: Extract and convert BRFSS 2023 data from XPT format
- **Input**: `data/LLCP2023.XPT` (raw BRFSS data)
- **Output**: 
  - `data/LLCP2023.csv` (converted dataset)
  - `data/LLCP2023_metadata.csv` (variable metadata)
- **Key Features**:
  - Handles encoding issues (latin1, cp1252)
  - Extracts variable labels and value mappings from HTML metadata
  - Creates comprehensive variable documentation

#### `02_data_preprocessing.py`
- **Purpose**: Clean and engineer features for causal analysis
- **Input**: `data/LLCP2023.csv`
- **Output**: `data/LLCP2023_processed.csv`
- **Key Features**:
  - Processes Adverse Childhood Experiences (ACE) variables
  - Creates composite indices for substance use, social determinants, healthcare access
  - Normalizes continuous variables
  - Handles missing data and special codes
  - Engineers exercise-related features

#### `03_propensity_score_matching.py`
- **Purpose**: Perform propensity score matching to create balanced groups
- **Input**: `data/LLCP2023_processed.csv`
- **Output**: 
  - `data/best_matched_llcp2023.csv` (matched dataset)
  - `data/matching_pairs.csv` (matching pairs)
  - `data/balance_before_matching.csv` (balance metrics)
  - `data/balance_after_matching.csv` (balance metrics)
  - `data/treatment_effect_results.csv` (treatment effect estimates)
- **Key Features**:
  - Defines treatment (exercise participation) and outcome (mental health) variables
  - Calculates propensity scores using logistic regression
  - Performs nearest neighbor matching with caliper
  - Assesses covariate balance before and after matching
  - Estimates average treatment effects

#### `03a_psm_specification_testing.py`
- **Purpose**: Test different PSM specifications to find optimal matching strategy
- **Input**: `data/LLCP2023_processed.csv`
- **Output**:
  - `data/enhanced_psm_specification_comparison.csv` (specification comparison)
  - `data/enhanced_psm_specification_summary.txt` (detailed summary)
- **Key Features**:
  - Tests multiple confounder specifications (minimal, standard, comprehensive, extended)
  - Compares balance quality, sample size, and treatment effects
  - Recommends optimal specification based on balance criteria

#### `04_causal_model_fitting.py`
- **Purpose**: Perform comprehensive causal effect analysis using Inverse Probability Weighting (IPW)
- **Input**: `data/LLCP2023_processed.csv`
- **Output**:
  - `data/causal_analysis_dataset_ipw.csv` (analysis dataset with IPW weights)
  - `data/ipw_dose_response_results.csv` (IPW dose-response analysis)
  - `data/ipw_heterogeneous_effects_results.csv` (IPW subgroup analysis)
  - `visualizations/causal_analysis_results_ipw.png` (comprehensive visualization)
- **Key Features**:
  - **CausalEffectAnalyzer Class**: Encapsulates entire analysis pipeline
  - **Exercise Measures**: Creates frequency, intensity, volume, and binary treatment variables
  - **Propensity Score Calculation**: Uses RandomForestClassifier with optimized parameters
  - **IPW Analysis**: Applies inverse probability weighting for causal inference
  - **Dose-Response Analysis**: Examines relationship between exercise minutes and mental health
  - **Heterogeneous Effects**: Analyzes treatment effects across subgroups (gender, income, baseline mental health)
  - **Robust Error Handling**: Handles edge cases like zero-sum weights and empty bins
  - **Optimized Visualization**: Creates multi-panel plots with automatic size adjustment
- **Recent Improvements** (August 2025):
  - Fixed structural issues (removed duplicate methods and functions)
  - Added comprehensive error handling for edge cases
  - Optimized RandomForest parameters for better performance
  - Improved visualization handling to prevent matplotlib size errors
  - Added weighted regression analysis with proper standard errors

#### `05_visualization.py`
- **Purpose**: Visualizes the results for blog
- **Input**: 
  - `data/balance_before_matching.csv`
  - `data/balance_after_matching.csv`
  - `data/treatment_effect_results.csv`
  - `data/ipw_dose_response_results.csv`

- **Output**:
  - `visualizations/overall_balance_improvement.png`
  - `visualizations/SMD_balance_improvement.png`
  - `visualizations/treatment_effect.png`
  - `visualizations/ipw_dose_response.png`
  - `visualizations/ipw_heterogeneous_effects.png`


- **Key Features**:
  - Creates Visualizations to measure the impact of PSM and specification engineering

### Data Directory

The `data/` directory contains:
- **Raw Data**: `LLCP2023.XPT`, `USCODE23_LLCP_021924.HTML`
- **Processed Data**: `LLCP2023.csv`, `LLCP2023_processed.csv`
- **Analysis Results**: Various CSV files with matching results, balance metrics, and treatment effects
- **Metadata**: `LLCP2023_metadata.csv`

### Visualizations Directory

The `visualizations/` directory contains:
- `psm_analysis_results.png` - Propensity score matching visualizations
- `enhanced_psm_specification_comparison.png` - PSM specification testing results
- `causal_analysis_results.png` - Causal effect analysis plots
- `visualizations/overall_balance_improvement.png` - Overall balance improvement from optimized PSM specification
- `visualizations/SMD_balance_improvement.png` - Feature-level SMD improvement from Optimized PSM specifications
- `visualizations/treatment_effect.png` - Average Treatment Effect
- `visualizations/ipw_dose_response.png` - IPW Dose-Response Curve
- `visualizations/ipw_heterogeneous_effects.png` - IPW Heterogeneous Effects

## Key Variables

### Treatment Variable
- **Exercise Participation**: Binary indicator based on strength training, aerobic exercise, or other physical activities

### Outcome Variable
- **Mental Health**: Binary indicator of good vs. poor mental health (≤15 days of poor mental health per month)

### Confounders
- **Demographics**: Age, sex, marital status, education, income, employment
- **Health Conditions**: Depression history, disability status
- **Substance Use**: Alcohol, smoking, marijuana use indices
- **Social Determinants**: Financial insecurity, food insecurity, emotional distress
- **Healthcare Access**: Insurance status, cost barriers, regular checkups
- **Geographic**: Metropolitan area, urban/rural classification
- **Risk Factors**: Adverse childhood experiences, sexual orientation

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd counterfactual_modeling_BRFSS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**:
   - Place BRFSS 2023 XPT file in `data/LLCP2023.XPT`
   - Place HTML metadata file in `data/USCODE23_LLCP_021924.HTML`

## Usage

### Complete Analysis Pipeline

Run the analysis in sequence:

```bash
# 1. Extract and convert data
python 01_data_extraction.py

# 2. Preprocess data
python 02_data_preprocessing.py

# 3a. Test PSM specifications (optional)
python 03a_psm_specification_testing.py

# 3b. Perform propensity score matching
python 03_propensity_score_matching.py

# 4. Fit causal models
python 04_causal_model_fitting.py

# 5. Visualize results
python 05_visualization.py
```

### Individual Components

Each script can be run independently if the required input files are available:

```python
# Example: Run PSM analysis
from propensity_score_matching import PropensityScoreMatcher

matcher = PropensityScoreMatcher()
matcher.run_full_analysis(caliper=0.2, ratio=1)
```

## Results Interpretation

### Propensity Score Matching
- **Balance Assessment**: Standardized mean differences (SMD) < 0.1 indicate good balance
- **Treatment Effect**: Average treatment effect (ATE) represents the causal effect of exercise on mental health
- **Sample Size**: Number of matched pairs affects statistical power

### Causal Analysis
- **Linear Effects**: Regression coefficients show magnitude and direction of effects
- **Dose-Response**: Non-linear relationships between exercise intensity and mental health
- **Heterogeneous Effects**: Treatment effects vary across demographic subgroups

## Technical Details

### Propensity Score Matching
- **Algorithm**: Nearest neighbor matching with caliper
- **Model**: Logistic regression with standardized features
- **Balance Metric**: Standardized mean difference (SMD)
- **Quality Threshold**: |SMD| < 0.1 for good balance

### Causal Inference Methods
- **Linear Regression**: Continuous outcome analysis
- **Dose-Response Analysis**: Non-linear effect estimation
- **Subgroup Analysis**: Heterogeneous treatment effects

## Dependencies

- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computing
- **scikit-learn** (≥1.1.0): Machine learning algorithms
- **matplotlib** (≥3.5.0): Data visualization
- **seaborn** (≥0.11.0): Statistical visualization
- **pyreadstat** (≥1.1.0): SAS/SPSS file reading
- **beautifulsoup4** (≥4.11.0): HTML parsing
- **statsmodels** (>=0.13.0): Statistical modeling and analysis



## Notes

- The analysis uses the 2023 BRFSS dataset, which is publicly available from the CDC
- All results are based on observational data and should be interpreted with appropriate caution
- The propensity score matching approach helps reduce confounding but cannot eliminate all bias
- Results may not be generalizable to populations not represented in the BRFSS sample
- The data/ directory has been removed from the repository. Download the raw files(`LLCP2023.XPT`, `USCODE23_LLCP_021924.HTML`) from https://www.cdc.gov/brfss/annual_data/annual_2023.html and place it under data/ directory to run the code.
