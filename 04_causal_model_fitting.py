#!/usr/bin/env python3
"""
Causal Model Fitting for Exercise Impact on Mental Health
Analyzes the causal effect of exercise frequency and intensity on mental health
using continuous variables and multiple causal inference methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CausalEffectAnalyzer:
    """Comprehensive causal effect analysis for exercise on mental health."""
    
    def __init__(self, processed_data_path='data/LLCP2023_processed.csv'):
        """Initialize the analyzer with processed data."""
        self.processed_data_path = processed_data_path
        self.df = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load processed data and prepare for causal analysis."""
        print("Loading processed dataset...")
        self.df = pd.read_csv(self.processed_data_path)
        
        # Add unique identifier if not present (from 03_propensity_score_matching.py)
        if '_SEQNO' not in self.df.columns:
            self.df['_SEQNO'] = range(len(self.df))
        
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Create comprehensive exercise measures
        self.create_exercise_measures()
        
        # Prepare mental health outcome
        self.prepare_mental_health_outcome()
        
        # Define confounders (same as PSM analysis)
        self.define_confounders()
        
        # Calculate propensity scores and IPW weights internally
        self.calculate_propensity_scores_internal()
        
        # Prepare analysis dataset
        self.prepare_analysis_dataset()
        
        return self
    
    def create_exercise_measures(self):
        """Create comprehensive exercise measures from available variables."""
        print("\n=== Creating Exercise Measures ===")
        
        # 1. Total exercise days per week
        self.df['TOTAL_EXERCISE_DAYS'] = (
            self.df['STRENGTH_WEEKLY'].fillna(0) + 
            self.df['EXEROFT1_WEEKLY'].fillna(0) + 
            self.df['EXEROFT2_WEEKLY'].fillna(0)
        )
        
        # 2. Exercise intensity (minutes per week)
        self.df['TOTAL_EXERCISE_MINUTES'] = (
            self.df['EXERHMM1_MIN'].fillna(0) + 
            self.df['EXERHMM2_MIN'].fillna(0)
        )
        
        # 3. Exercise frequency (days per week, capped at 7)
        self.df['EXERCISE_FREQUENCY'] = np.minimum(self.df['TOTAL_EXERCISE_DAYS'], 7)
        
        # 4. Exercise intensity per session (average minutes per session)
        total_sessions = (
            (self.df['STRENGTH_WEEKLY'] > 0).astype(int) + 
            (self.df['EXEROFT1_WEEKLY'] > 0).astype(int) + 
            (self.df['EXEROFT2_WEEKLY'] > 0).astype(int)
        )
        self.df['EXERCISE_INTENSITY_PER_SESSION'] = np.where(
            total_sessions > 0,
            self.df['TOTAL_EXERCISE_MINUTES'] / total_sessions,
            0
        )
        
        # 5. Exercise volume (frequency × intensity)
        self.df['EXERCISE_VOLUME'] = self.df['EXERCISE_FREQUENCY'] * self.df['EXERCISE_INTENSITY_PER_SESSION']
        
        # 6. Exercise categories
        self.df['EXERCISE_CATEGORY'] = pd.cut(
            self.df['EXERCISE_FREQUENCY'],
            bins=[-1, 0, 2, 4, 7],
            labels=['None', 'Low', 'Moderate', 'High']
        )
        
        # Add _EXERCISE_BINARY (from 03_propensity_score_matching.py)
        self.df['_EXERCISE_BINARY'] = np.where(
            (self.df['STRENGTH_WEEKLY'] > 0) |
            (self.df['EXEROFT1_WEEKLY'] > 0) |
            (self.df['EXEROFT2_WEEKLY'] > 0),
            1, 0
        )

        print("Exercise measures created:")
        print(f"- Exercise Frequency: {self.df['EXERCISE_FREQUENCY'].describe()}")
        print(f"- Exercise Intensity: {self.df['EXERCISE_INTENSITY_PER_SESSION'].describe()}")
        print(f"- Exercise Volume: {self.df['EXERCISE_VOLUME'].describe()}")
        print(f"- Exercise Binary (Treatment): {self.df['_EXERCISE_BINARY'].value_counts()}")

        return self

    def prepare_mental_health_outcome(self):
        """Prepare mental health outcome variable."""
        print("\n=== Preparing Mental Health Outcome ===")

        # Use the continuous mental health measure
        self.df['MENTAL_HEALTH_CONTINUOUS'] = self.df['MENTHLTH_MAPPED']

        # Create mental health categories for analysis
        self.df['MENTAL_HEALTH_CATEGORY'] = pd.cut(
            self.df['MENTAL_HEALTH_CONTINUOUS'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Poor', 'Fair', 'Good']
        )

        print(f"Mental health outcome prepared:")
        print(f"- Continuous measure: {self.df['MENTAL_HEALTH_CONTINUOUS'].describe()}")
        print(f"- Categories: {self.df['MENTAL_HEALTH_CATEGORY'].value_counts()}")

        return self

    def define_confounders(self):
        """Define confounders (same as PSM analysis)."""
        # Define optimal confounders (full_model specification from enhanced testing)
        self.confounders = [
            # Demographics
            'SEXVAR', 'MARITAL', 'EDUCA', 'INCOME3', 'EMPLOY1', 'ADDEPEV3',
            'DISABILITY_INDEX_NORMALIZED', 'WEIGHT2', 'HEIGHT3', 'CHILDREN',

            # Substance use
            'ALCOHOL_SEVERITY_NORMALIZED', 'SMOKING_INDEX_NORMALIZED',
            'MARIJUANA_INDEX_NORMALIZED',

            # Social determinants
            'FINANCIAL_INSECURITY_INDEX_NORMALIZED', 'FOOD_INSECURITY_INDEX_NORMALIZED',
            'EMOTIONAL_DISTRESS_INDEX_NORMALIZED', 'FACING_UNEMPLOYMENT_INDEX_NORMALIZED',

            # Healthcare access
            'HEALTHCAREACCESS_INDEX_NORMALIZED',

            # Geographic
            'METRO_AREA', 'URBAN_RURAL_AREA',

            # Other risk factors
            'ACE_NORMALIZED', 'SEXUAL_ORIENTATION_SCORE', 'VETERAN3', 'RENTHOM1', 'PREGNANT'
        ]

        # Filter to only existing columns
        self.confounders = [col for col in self.confounders if col in self.df.columns]

        return self

    def prepare_analysis_dataset(self):
        """Prepare final analysis dataset."""
        print("\n=== Preparing Analysis Dataset ===")

        # Select variables for analysis
        analysis_vars = (
            ['EXERCISE_FREQUENCY', 'EXERCISE_INTENSITY_PER_SESSION', 'EXERCISE_VOLUME',
             'MENTAL_HEALTH_CONTINUOUS', 'MENTAL_HEALTH_CATEGORY',
             '_EXERCISE_BINARY', 'propensity_score', 'ipw_weight'] +
            self.confounders
        )

        # Filter for complete cases
        self.df_analysis = self.df.dropna(subset=analysis_vars).copy()

        print(f"Analysis dataset: {self.df_analysis.shape[0]} rows")

        # Convert categorical variables to numeric codes
        categorical_vars = ['SEXVAR', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3',
                           'EMPLOY1', 'CHILDREN', 'INCOME3', 'PREGNANT',
                           'METRO_AREA', 'URBAN_RURAL_AREA', 'ADDEPEV3']

        for var in categorical_vars:
            if var in self.confounders:
                self.df_analysis[var] = pd.Categorical(self.df_analysis[var]).codes

        # Ensure all confounders are numeric
        for var in self.confounders:
            self.df_analysis[var] = pd.to_numeric(self.df_analysis[var], errors='coerce')

        # Remove any remaining NaN values
        self.df_analysis = self.df_analysis.dropna(subset=self.confounders)

        print(f"Final analysis dataset: {self.df_analysis.shape[0]} rows")

        return self

    def calculate_propensity_scores_internal(self):
        """Calculate propensity scores and IPW weights internally."""
        print("\n=== Calculating Propensity Scores Internally ===")

        # Prepare data for PS calculation (similar to 03_propensity_score_matching.py)
        # Select analysis variables for PS calculation
        ps_analysis_cols = ['_SEQNO', '_EXERCISE_BINARY'] + self.confounders
        ps_df = self.df[ps_analysis_cols].dropna().copy()

        # Convert categorical variables to numeric codes for PS calculation
        categorical_vars_ps = ['SEXVAR', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3',
                               'EMPLOY1', 'CHILDREN', 'INCOME3', 'PREGNANT',
                               'METRO_AREA', 'URBAN_RURAL_AREA', 'ADDEPEV3']

        for var in categorical_vars_ps:
            if var in ps_df.columns:
                ps_df[var] = pd.Categorical(ps_df[var]).codes
                ps_df[var] = ps_df[var].replace(-1, np.nan) # Replace -1 (NaN) with actual NaN

        ps_df.dropna(inplace=True) # Drop any remaining NaNs after conversion

        X_ps = ps_df[self.confounders]
        y_ps = ps_df['_EXERCISE_BINARY']

        # Standardize features for PS model
        scaler_ps = StandardScaler()
        X_ps_scaled = scaler_ps.fit_transform(X_ps)

        # Use Random Forest (optimal model from 03_propensity_score_matching.py)
        # Reduced n_estimators for faster computation while maintaining accuracy
        ps_model = RandomForestClassifier(
            n_estimators=50,  # Reduced from 100 for better performance
            random_state=42,
            max_depth=5,
            n_jobs=-1,  # Use all available cores
            min_samples_split=20,  # Prevent overfitting
            min_samples_leaf=10  # Prevent overfitting
        )
        ps_model.fit(X_ps_scaled, y_ps)

        # Calculate propensity scores
        ps_df['propensity_score'] = ps_model.predict_proba(X_ps_scaled)[:, 1]

        # Merge propensity scores back to the main dataframe
        self.df = pd.merge(self.df, ps_df[['_SEQNO', 'propensity_score']], on='_SEQNO', how='left')

        # Calculate IPW weights
        # ATE weights: w = T/PS + (1-T)/(1-PS)
        self.df['ipw_weight'] = np.where(self.df['_EXERCISE_BINARY'] == 1,
                                         1 / self.df['propensity_score'],
                                         1 / (1 - self.df['propensity_score']))

        # Handle potential infinite weights (e.g., if PS is 0 or 1)
        self.df['ipw_weight'] = self.df['ipw_weight'].replace([np.inf, -np.inf], np.nan)
        self.df.dropna(subset=['ipw_weight'], inplace=True)

        print(f"Propensity scores and IPW weights calculated. Dataset size after dropping NaNs: {self.df.shape[0]} rows")

        return self

    def linear_regression_analysis(self):
        """This method is replaced by IPW analysis."""
        pass

    def ipw_dose_response_analysis(self):
        """Analyze dose-response relationship between exercise minutes and mental health using IPW."""
        print("\n=== IPW Dose-Response Analysis ===")

        # Use statsmodels for weighted linear regression

        # Define the model formula
        # We are interested in the effect of TOTAL_EXERCISE_MINUTES on MENTAL_HEALTH_CONTINUOUS
        # The IPW weights account for confounding
        formula = 'MENTAL_HEALTH_CONTINUOUS ~ TOTAL_EXERCISE_MINUTES'

        # Fit the weighted linear regression model
        model = smf.wls(formula, data=self.df_analysis, weights=self.df_analysis['ipw_weight'])
        results = model.fit()

        print(results.summary())

        # For dose-response curve, we can predict mental health for a range of exercise minutes
        # and plot the weighted average mental health for bins of exercise minutes

        # Create exercise minutes bins with edge case handling
        max_minutes = self.df_analysis['TOTAL_EXERCISE_MINUTES'].max()
        if max_minutes > 0:
            bins = np.arange(0, max_minutes + 60, 60)
            labels = [f'{i}-{i+59}' for i in np.arange(0, max_minutes, 60)]
            self.df_analysis['EXERCISE_MINUTES_BINS'] = pd.cut(
                self.df_analysis['TOTAL_EXERCISE_MINUTES'],
                bins=bins,
                right=False,  # Include left, exclude right
                labels=labels
            )
        else:
            self.df_analysis['EXERCISE_MINUTES_BINS'] = '0-59'

        # Calculate weighted mean mental health by exercise minutes bins
        # Use a custom aggregation function for weighted mean and standard error
        def weighted_mean_se(group):
            # Check if weights are valid
            if len(group) == 0 or group['ipw_weight'].sum() == 0:
                return pd.Series({'mean': np.nan, 'se': np.nan, 'count': 0})
            
            try:
                mean = np.average(group['MENTAL_HEALTH_CONTINUOUS'], weights=group['ipw_weight'])
                # For weighted standard error, a more complex formula is needed,
                # but for simplicity, we can use unweighted std for visualization purposes
                # or a robust estimator if available in statsmodels.
                # For now, let's use a simple weighted std for error bars.
                variance = np.average((group['MENTAL_HEALTH_CONTINUOUS'] - mean)**2, weights=group['ipw_weight'])
                std_err = np.sqrt(variance / len(group)) # Approximation
                return pd.Series({'mean': mean, 'se': std_err, 'count': len(group)})
            except (ZeroDivisionError, ValueError) as e:
                print(f"  Warning: Error calculating weighted mean: {e}")
                return pd.Series({'mean': np.nan, 'se': np.nan, 'count': len(group)})

        # Only calculate dose response if we have valid bins
        if 'EXERCISE_MINUTES_BINS' in self.df_analysis.columns and not self.df_analysis['EXERCISE_MINUTES_BINS'].isna().all():
            dose_response = self.df_analysis.groupby('EXERCISE_MINUTES_BINS').apply(weighted_mean_se).reset_index()
            # Remove rows with NaN values to avoid plotting issues
            dose_response = dose_response.dropna(subset=['mean', 'se'])
        else:
            dose_response = pd.DataFrame()
        
        if not dose_response.empty:
            dose_response['ci_lower'] = dose_response['mean'] - 1.96 * dose_response['se']
            dose_response['ci_upper'] = dose_response['mean'] + 1.96 * dose_response['se']

        self.results['ipw_dose_response'] = dose_response
        self.results['ipw_dose_response_model'] = results

        print("IPW Dose-response relationship:")
        print(dose_response)

        return self

    def ipw_heterogeneous_effects_analysis(self):
        """Analyze heterogeneous treatment effects across subgroups using IPW."""
        print("\n=== IPW Heterogeneous Effects Analysis ===")


        # Define subgroups
        subgroups = {
            'Gender': self.df_analysis['SEXVAR'].map({0: 'Male', 1: 'Female'}),
            'Income': pd.cut(self.df_analysis['INCOME3'], bins=3, labels=['Low', 'Middle', 'High']),
            'Mental Health Baseline': pd.cut(
                self.df_analysis['MENTAL_HEALTH_CONTINUOUS'],
                bins=3, labels=['Poor', 'Fair', 'Good']
            )
        }

        heterogeneous_results = {}

        for subgroup_name, subgroup_labels in subgroups.items():
            print(f"\nAnalyzing subgroup: {subgroup_name}")

            subgroup_effects = []

            for label in subgroup_labels.unique():
                if pd.isna(label):
                    continue

                mask = subgroup_labels == label
                subgroup_data = self.df_analysis[mask].copy() # Use .copy() to avoid SettingWithCopyWarning

                if len(subgroup_data) < 50:  # Skip small subgroups
                    print(f"  Skipping {label}: sample size too small ({len(subgroup_data)})")
                    continue

                # Weighted linear regression for this subgroup with error handling
                # We are interested in the effect of _EXERCISE_BINARY
                formula = 'MENTAL_HEALTH_CONTINUOUS ~ _EXERCISE_BINARY'

                try:
                    model = smf.wls(formula, data=subgroup_data, weights=subgroup_data['ipw_weight'])
                    results = model.fit()
                except Exception as e:
                    print(f"  Error fitting model for {label}: {e}")
                    continue

                # Extract coefficient for _EXERCISE_BINARY
                if '_EXERCISE_BINARY' in results.params:
                    exercise_coef = results.params['_EXERCISE_BINARY']
                    subgroup_effects.append({
                        'subgroup': label,
                        'effect': exercise_coef,
                        'sample_size': len(subgroup_data)
                    })
                    print(f"  {label}: {exercise_coef:.4f} (n={len(subgroup_data)})")
                else:
                    print(f"  Could not find '_EXERCISE_BINARY' coefficient for {label}")

            heterogeneous_results[subgroup_name] = subgroup_effects

        self.results['ipw_heterogeneous_effects'] = heterogeneous_results
        return self

    def create_visualizations(self):
        """Create comprehensive visualization of results."""
        print("\n=== Creating Visualizations ===")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Causal Effect Analysis Results (IPW)', fontsize=16, fontweight='bold')

        # 1. Exercise Distribution (TOTAL_EXERCISE_MINUTES)
        ax1 = axes[0, 0]
        self.df_analysis['TOTAL_EXERCISE_MINUTES'].hist(bins=20, alpha=0.7, ax=ax1)
        ax1.set_xlabel('Exercise Minutes per Week')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Exercise Minutes')
        ax1.grid(True, alpha=0.3)

        # 2. Mental Health Distribution
        ax2 = axes[0, 1]
        self.df_analysis['MENTAL_HEALTH_CONTINUOUS'].hist(bins=20, alpha=0.7, ax=ax2)
        ax2.set_xlabel('Mental Health Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Mental Health')
        ax2.grid(True, alpha=0.3)

        # 3. Scatter Plot: Exercise Minutes vs Mental Health (Weighted)
        ax3 = axes[0, 2]
        # For weighted scatter plot, we can use size of points proportional to weights or just plot
        # For simplicity, let's plot and indicate it's based on weighted data.
        ax3.scatter(self.df_analysis['TOTAL_EXERCISE_MINUTES'],
                   self.df_analysis['MENTAL_HEALTH_CONTINUOUS'],
                   alpha=0.6, s=20)

        # Add weighted trend line from IPW dose-response model
        if 'ipw_dose_response_model' in self.results:
            model_results = self.results['ipw_dose_response_model']
            min_ex = self.df_analysis['TOTAL_EXERCISE_MINUTES'].min()
            max_ex = self.df_analysis['TOTAL_EXERCISE_MINUTES'].max()
            x_vals = np.linspace(min_ex, max_ex, 100)
            # Create a dummy dataframe for prediction
            pred_df = pd.DataFrame({'TOTAL_EXERCISE_MINUTES': x_vals})
            y_pred = model_results.predict(pred_df)
            ax3.plot(x_vals, y_pred, "r--", alpha=0.8, label='IPW Weighted Trend')
            ax3.legend()

        ax3.set_xlabel('Exercise Minutes per Week')
        ax3.set_ylabel('Mental Health Score')
        ax3.set_title('Exercise Minutes vs Mental Health (Weighted)')
        ax3.grid(True, alpha=0.3)

        # 4. IPW Dose-Response Relationship
        ax4 = axes[1, 0]
        # Dose-Response Curve
        dose_data = self.results.get('ipw_dose_response', pd.DataFrame())
        
        if not dose_data.empty:
            # Filter out bins with too few observations for cleaner plot
            dose_data = dose_data[dose_data['count'] > 10] # Example threshold
            
            # Sort by exercise minutes and limit to first 20 bins to avoid plotting issues
            # First, extract the lower bound of each bin for sorting
            try:
                dose_data['bin_start'] = dose_data['EXERCISE_MINUTES_BINS'].astype(str).str.split('-').str[0].astype(float)
                dose_data = dose_data.sort_values('bin_start')
                dose_data = dose_data.drop('bin_start', axis=1)
            except:
                pass  # If conversion fails, keep original order
            
            # Limit to first 20 bins
            if len(dose_data) > 20:
                dose_data = dose_data.iloc[:20]
                ax4.set_title('IPW Dose-Response Curve (First 20 bins)', fontsize=12, fontweight='bold')
            else:
                ax4.set_title('IPW Dose-Response Curve', fontsize=12, fontweight='bold')
        else:
            ax4.set_title('IPW Dose-Response Curve', fontsize=12, fontweight='bold')

        if not dose_data.empty:
            # Convert bin labels to numeric for plotting with error handling
            # Assuming labels are like '0-59', '60-119' etc. take the lower bound
            try:
                plot_x = [int(str(l).split('-')[0]) for l in dose_data['EXERCISE_MINUTES_BINS']]
            except (ValueError, AttributeError):
                # If conversion fails, use index
                plot_x = range(len(dose_data))

            ax4.errorbar(plot_x, dose_data['mean'],
                        yerr=dose_data['se'], fmt='o-', capsize=5)
            ax4.set_xlabel('Exercise Minutes per Week')
            ax4.set_ylabel('Weighted Mean Mental Health Score')
            ax4.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if there are many bins
            if len(plot_x) > 10:
                ax4.tick_params(axis='x', rotation=45)
            
            # Limit number of x-axis ticks to prevent overcrowding
            ax4.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        else:
            ax4.text(0.5, 0.5, 'No dose-response data available',
                    ha='center', va='center', transform=ax4.transAxes)

        # 5. IPW Heterogeneous Effects (Example: Gender)
        ax5 = axes[1, 1]
        if 'ipw_heterogeneous_effects' in self.results:
            gender_effects = self.results['ipw_heterogeneous_effects'].get('Gender', [])
            if gender_effects:
                genders = [effect['subgroup'] for effect in gender_effects]
                effects = [effect['effect'] for effect in gender_effects]

                colors = ['red' if x < 0 else 'blue' for x in effects]
                bars = ax5.bar(genders, effects, color=colors, alpha=0.7)
                ax5.set_ylabel('IPW Exercise Effect')
                ax5.set_title('IPW Heterogeneous Effects by Gender')
                ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax5.grid(True, alpha=0.3)

                # Add value labels
                for bar, coef in zip(bars, effects):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'{coef:.4f}', ha='center', va='bottom', fontsize=8)

        # 6. Placeholder for another heterogeneous effect or overall summary
        ax6 = axes[1, 2]
        ax6.text(0.5, 0.5, 'Additional IPW Analysis / Summary',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax6.transAxes, fontsize=12, color='gray')
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_frame_on(False)

        # Use constrained layout instead of tight_layout to avoid size issues
        plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
        
        # Save with lower DPI if needed to avoid size issues
        try:
            plt.savefig('visualizations/causal_analysis_results_ipw.png', dpi=150, bbox_inches='tight')
        except ValueError as e:
            print(f"Warning: Could not save high-resolution image due to size constraints: {e}")
            # Try saving with even lower DPI
            plt.savefig('visualizations/causal_analysis_results_ipw.png', dpi=100)
        
        plt.show()

        return self

    def save_results(self):
        """Save all analysis results."""
        print("\n=== Saving Results ===")

        # Save main analysis dataset (now with IPW weights)
        self.df_analysis.to_csv('data/causal_analysis_dataset_ipw.csv', index=False)
        print("Causal analysis dataset (with IPW weights) saved to: data/causal_analysis_dataset_ipw.csv")

        # Save IPW dose-response results
        if 'ipw_dose_response' in self.results:
            self.results['ipw_dose_response'].to_csv('data/ipw_dose_response_results.csv', index=False)
            print("IPW Dose-response results saved to: data/ipw_dose_response_results.csv")

        # Save IPW heterogeneous effects
        if 'ipw_heterogeneous_effects' in self.results:
            het_results = []
            for subgroup, effects in self.results['ipw_heterogeneous_effects'].items():
                for effect in effects:
                    het_results.append({
                        'subgroup': subgroup,
                        'category': effect['subgroup'],
                        'effect': effect['effect'],
                        'sample_size': effect['sample_size']
                    })

            het_df = pd.DataFrame(het_results)
            het_df.to_csv('data/ipw_heterogeneous_effects_results.csv', index=False)
            print("IPW Heterogeneous effects results saved to: data/ipw_heterogeneous_effects_results.csv")

        return self

    def run_full_analysis(self):
        """Run the complete causal effect analysis."""
        print("=" * 60)
        print("CAUSAL EFFECT ANALYSIS (IPW)")
        print("Exercise Impact on Mental Health (Continuous Variables)")
        print("=" * 60)

        (self.load_and_prepare_data()
         .ipw_dose_response_analysis()
         .ipw_heterogeneous_effects_analysis()
         .create_visualizations()
         .save_results())

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED")
        print("=" * 60)

        return self

def main():
    """Main execution function."""
    # Initialize and run analysis
    analyzer = CausalEffectAnalyzer()
    analyzer.run_full_analysis()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
