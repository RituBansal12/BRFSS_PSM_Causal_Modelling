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

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CausalEffectAnalyzer:
    """Comprehensive causal effect analysis for exercise on mental health."""
    
    def __init__(self, data_path='data/LLCP2023_processed.csv'):
        """Initialize the analyzer with processed data."""
        self.data_path = data_path
        self.df = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load processed data and prepare for causal analysis."""
        print("Loading processed dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Create comprehensive exercise measures
        self.create_exercise_measures()
        
        # Prepare mental health outcome
        self.prepare_mental_health_outcome()
        
        # Define confounders (same as PSM analysis)
        self.define_confounders()
        
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
        
        print("Exercise measures created:")
        print(f"- Exercise Frequency: {self.df['EXERCISE_FREQUENCY'].describe()}")
        print(f"- Exercise Intensity: {self.df['EXERCISE_INTENSITY_PER_SESSION'].describe()}")
        print(f"- Exercise Volume: {self.df['EXERCISE_VOLUME'].describe()}")
        
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
             'MENTAL_HEALTH_CONTINUOUS', 'MENTAL_HEALTH_CATEGORY'] + 
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
    
    def linear_regression_analysis(self):
        """Perform linear regression analysis with confounder adjustment."""
        print("\n=== Linear Regression Analysis ===")
        
        # Prepare variables
        X = self.df_analysis[self.confounders]
        y = self.df_analysis['MENTAL_HEALTH_CONTINUOUS']
        
        # Standardize confounders
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Add exercise variables
        exercise_vars = ['EXERCISE_FREQUENCY', 'EXERCISE_INTENSITY_PER_SESSION', 'EXERCISE_VOLUME']
        
        results = {}
        
        for exercise_var in exercise_vars:
            print(f"\nAnalyzing: {exercise_var}")
            
            # Add exercise variable to features
            X_with_exercise = X_scaled.copy()
            X_with_exercise[exercise_var] = self.df_analysis[exercise_var]
            
            # Fit model
            model = LinearRegression()
            model.fit(X_with_exercise, y)
            
            # Predictions
            y_pred = model.predict(X_with_exercise)
            
            # Model performance
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # Extract exercise coefficient
            exercise_idx = X_with_exercise.columns.get_loc(exercise_var)
            exercise_coef = model.coef_[exercise_idx]
            exercise_std = np.std(X_with_exercise[exercise_var])
            
            # Calculate standardized effect
            standardized_effect = exercise_coef * exercise_std
            
            results[exercise_var] = {
                'coefficient': exercise_coef,
                'standardized_effect': standardized_effect,
                'r2': r2,
                'mse': mse,
                'model': model
            }
            
            print(f"  Coefficient: {exercise_coef:.4f}")
            print(f"  Standardized Effect: {standardized_effect:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MSE: {mse:.4f}")
        
        self.results['linear_regression'] = results
        return self
    
    def dose_response_analysis(self):
        """Analyze dose-response relationship between exercise and mental health."""
        print("\n=== Dose-Response Analysis ===")
        
        # Create exercise frequency bins
        self.df_analysis['EXERCISE_FREQ_BINS'] = pd.cut(
            self.df_analysis['EXERCISE_FREQUENCY'],
            bins=[-1, 0, 1, 2, 3, 4, 5, 6, 7],
            labels=['0', '1', '2', '3', '4', '5', '6', '7+']
        )
        
        # Calculate mean mental health by exercise frequency
        dose_response = self.df_analysis.groupby('EXERCISE_FREQ_BINS')['MENTAL_HEALTH_CONTINUOUS'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        dose_response['se'] = dose_response['std'] / np.sqrt(dose_response['count'])
        dose_response['ci_lower'] = dose_response['mean'] - 1.96 * dose_response['se']
        dose_response['ci_upper'] = dose_response['mean'] + 1.96 * dose_response['se']
        
        self.results['dose_response'] = dose_response
        
        print("Dose-response relationship:")
        print(dose_response)
        
        return self
    
    def heterogeneous_effects_analysis(self):
        """Analyze heterogeneous treatment effects across subgroups."""
        print("\n=== Heterogeneous Effects Analysis ===")
        
        # Define subgroups
        subgroups = {
            'Age': pd.cut(self.df_analysis['WEIGHT2'], bins=3, labels=['Young', 'Middle', 'Older']),
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
                subgroup_data = self.df_analysis[mask]
                
                if len(subgroup_data) < 50:  # Skip small subgroups
                    continue
                
                # Linear regression for this subgroup
                X_sub = subgroup_data[self.confounders]
                y_sub = subgroup_data['MENTAL_HEALTH_CONTINUOUS']
                exercise_sub = subgroup_data['EXERCISE_FREQUENCY']
                
                # Standardize
                scaler = StandardScaler()
                X_sub_scaled = scaler.fit_transform(X_sub)
                X_sub_scaled = pd.DataFrame(X_sub_scaled, columns=X_sub.columns, index=X_sub.index)
                X_sub_scaled['EXERCISE_FREQUENCY'] = exercise_sub
                
                # Fit model
                model = LinearRegression()
                model.fit(X_sub_scaled, y_sub)
                
                # Extract exercise coefficient
                exercise_coef = model.coef_[-1]  # Last coefficient is exercise
                
                subgroup_effects.append({
                    'subgroup': label,
                    'effect': exercise_coef,
                    'sample_size': len(subgroup_data)
                })
                
                print(f"  {label}: {exercise_coef:.4f} (n={len(subgroup_data)})")
            
            heterogeneous_results[subgroup_name] = subgroup_effects
        
        self.results['heterogeneous_effects'] = heterogeneous_results
        return self
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Causal Effect Analysis: Exercise Impact on Mental Health', 
                     fontsize=16, fontweight='bold')
        
        # 1. Exercise Distribution
        ax1 = axes[0, 0]
        self.df_analysis['EXERCISE_FREQUENCY'].hist(bins=20, alpha=0.7, ax=ax1)
        ax1.set_xlabel('Exercise Days per Week')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Exercise Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Mental Health Distribution
        ax2 = axes[0, 1]
        self.df_analysis['MENTAL_HEALTH_CONTINUOUS'].hist(bins=20, alpha=0.7, ax=ax2)
        ax2.set_xlabel('Mental Health Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Mental Health')
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter Plot: Exercise vs Mental Health
        ax3 = axes[0, 2]
        ax3.scatter(self.df_analysis['EXERCISE_FREQUENCY'], 
                   self.df_analysis['MENTAL_HEALTH_CONTINUOUS'], 
                   alpha=0.6, s=20)
        
        # Add trend line
        z = np.polyfit(self.df_analysis['EXERCISE_FREQUENCY'], 
                      self.df_analysis['MENTAL_HEALTH_CONTINUOUS'], 1)
        p = np.poly1d(z)
        ax3.plot(self.df_analysis['EXERCISE_FREQUENCY'], 
                p(self.df_analysis['EXERCISE_FREQUENCY']), "r--", alpha=0.8)
        
        ax3.set_xlabel('Exercise Days per Week')
        ax3.set_ylabel('Mental Health Score')
        ax3.set_title('Exercise vs Mental Health')
        ax3.grid(True, alpha=0.3)
        
        # 4. Dose-Response Relationship
        ax4 = axes[1, 0]
        if 'dose_response' in self.results:
            dose_data = self.results['dose_response']
            ax4.errorbar(range(len(dose_data)), dose_data['mean'], 
                        yerr=dose_data['se'], fmt='o-', capsize=5)
            ax4.set_xlabel('Exercise Days per Week')
            ax4.set_ylabel('Mental Health Score')
            ax4.set_title('Dose-Response Relationship')
            ax4.set_xticks(range(len(dose_data)))
            ax4.set_xticklabels(dose_data['EXERCISE_FREQ_BINS'])
            ax4.grid(True, alpha=0.3)
        
        # 5. Linear Regression Results
        ax5 = axes[1, 1]
        if 'linear_regression' in self.results:
            exercise_vars = list(self.results['linear_regression'].keys())
            coefficients = [self.results['linear_regression'][var]['coefficient'] 
                          for var in exercise_vars]
            
            colors = ['red' if x < 0 else 'blue' for x in coefficients]
            bars = ax5.bar(exercise_vars, coefficients, color=colors, alpha=0.7)
            ax5.set_ylabel('Coefficient')
            ax5.set_title('Exercise Effects on Mental Health')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax5.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, coef in zip(bars, coefficients):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{coef:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Heterogeneous Effects
        ax6 = axes[1, 2]
        if 'heterogeneous_effects' in self.results:
            # Plot gender effects as example
            gender_effects = self.results['heterogeneous_effects'].get('Gender', [])
            if gender_effects:
                genders = [effect['subgroup'] for effect in gender_effects]
                effects = [effect['effect'] for effect in gender_effects]
                
                colors = ['red' if x < 0 else 'blue' for x in effects]
                bars = ax6.bar(genders, effects, color=colors, alpha=0.7)
                ax6.set_ylabel('Exercise Effect')
                ax6.set_title('Heterogeneous Effects by Gender')
                ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/causal_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def save_results(self):
        """Save all analysis results."""
        print("\n=== Saving Results ===")
        
        # Save main analysis dataset
        self.df_analysis.to_csv('data/causal_analysis_dataset.csv', index=False)
        print("Causal analysis dataset saved to: data/causal_analysis_dataset.csv")
        
        # Save dose-response results
        if 'dose_response' in self.results:
            self.results['dose_response'].to_csv('data/dose_response_results.csv', index=False)
            print("Dose-response results saved to: data/dose_response_results.csv")
        
        # Save linear regression results
        if 'linear_regression' in self.results:
            lr_results = []
            for var, result in self.results['linear_regression'].items():
                lr_results.append({
                    'exercise_variable': var,
                    'coefficient': result['coefficient'],
                    'standardized_effect': result['standardized_effect'],
                    'r2': result['r2'],
                    'mse': result['mse']
                })
            
            lr_df = pd.DataFrame(lr_results)
            lr_df.to_csv('data/linear_regression_results.csv', index=False)
            print("Linear regression results saved to: data/linear_regression_results.csv")
        
        # Save heterogeneous effects
        if 'heterogeneous_effects' in self.results:
            het_results = []
            for subgroup, effects in self.results['heterogeneous_effects'].items():
                for effect in effects:
                    het_results.append({
                        'subgroup': subgroup,
                        'category': effect['subgroup'],
                        'effect': effect['effect'],
                        'sample_size': effect['sample_size']
                    })
            
            het_df = pd.DataFrame(het_results)
            het_df.to_csv('data/heterogeneous_effects_results.csv', index=False)
            print("Heterogeneous effects results saved to: data/heterogeneous_effects_results.csv")
        
        return self
    
    def run_full_analysis(self):
        """Run the complete causal effect analysis."""
        print("=" * 60)
        print("CAUSAL EFFECT ANALYSIS")
        print("Exercise Impact on Mental Health (Continuous Variables)")
        print("=" * 60)
        
        (self.load_and_prepare_data()
         .linear_regression_analysis()
         .dose_response_analysis()
         .heterogeneous_effects_analysis()
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
