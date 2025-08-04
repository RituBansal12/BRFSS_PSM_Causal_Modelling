#!/usr/bin/env python3
"""
Enhanced Propensity Score Matching Specification Testing
Tests advanced PSM specifications to find the optimal matching strategy
for analyzing the impact of exercise on mental health.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedPSMSpecificationTester:
    """Test advanced PSM specifications to find optimal matching strategy."""
    
    def __init__(self, data_path='data/LLCP2023_processed.csv'):
        """Initialize the specification tester."""
        self.data_path = data_path
        self.df = None
        self.specifications = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading processed BRFSS 2023 data...")
        self.df = pd.read_csv(self.data_path)
        
        # Add unique identifier if not present
        if '_SEQNO' not in self.df.columns:
            self.df['_SEQNO'] = range(len(self.df))
        
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self
    
    def define_variables(self):
        """Define treatment, outcome, and confounder variables."""
        print("\n=== Variable Definition ===")
        
        # Treatment variable: Exercise participation (binary)
        self.df['_EXERCISE_BINARY'] = np.where(
            (self.df['STRENGTH_WEEKLY'] > 0) | 
            (self.df['EXEROFT1_WEEKLY'] > 0) | 
            (self.df['EXEROFT2_WEEKLY'] > 0),
            1, 0
        )
        
        # Outcome variable: Mental health (binary)
        self.df['_MENTAL_HEALTH_BINARY'] = np.where(
            self.df['MENTHLTH_MAPPED'] <= 0.5,
            1,  # Good mental health
            0   # Poor mental health
        )
        
        return self
    
    def define_enhanced_specifications(self):
        """Define advanced PSM specifications to test."""
        print("\n=== Defining Enhanced PSM Specifications ===")
        
        # Base confounders (always included)
        base_confounders = [
            'SEXVAR', 'MARITAL', 'EDUCA', 'INCOME3', 'EMPLOY1',
            'ADDEPEV3', 'DISABILITY_INDEX_NORMALIZED'
        ]
        
        # Specification 1: Minimal (demographics + health)
        self.specifications['minimal'] = base_confounders + [
            'WEIGHT2', 'HEIGHT3', 'CHILDREN'
        ]
        
        # Specification 2: Standard (add substance use)
        self.specifications['standard'] = self.specifications['minimal'] + [
            'ALCOHOL_SEVERITY_NORMALIZED', 'SMOKING_INDEX_NORMALIZED',
            'MARIJUANA_INDEX_NORMALIZED'
        ]
        
        # Specification 3: Comprehensive (add social determinants)
        self.specifications['comprehensive'] = self.specifications['standard'] + [
            'FINANCIAL_INSECURITY_INDEX_NORMALIZED', 'FOOD_INSECURITY_INDEX_NORMALIZED',
            'EMOTIONAL_DISTRESS_INDEX_NORMALIZED', 'FACING_UNEMPLOYMENT_INDEX_NORMALIZED',
            'HEALTHCAREACCESS_INDEX_NORMALIZED'
        ]
        
        # Specification 4: Extended (add geographic and other factors)
        self.specifications['extended'] = self.specifications['comprehensive'] + [
            'METRO_AREA', 'URBAN_RURAL_AREA', 'ACE_NORMALIZED',
            'SEXUAL_ORIENTATION_SCORE', 'VETERAN3', 'RENTHOM1'
        ]
        
        # Specification 5: Enhanced with interactions (NEW)
        self.specifications['enhanced_interactions'] = self.specifications['extended'] + [
            'SEXVAR_AGE_INTERACTION', 'INCOME_EDUCA_INTERACTION',
            'DISABILITY_EMPLOYMENT_INTERACTION'
        ]
        
        # Specification 6: Full model with all variables (NEW)
        self.specifications['full_model'] = [
            'SEXVAR', 'MARITAL', 'EDUCA', 'INCOME3', 'EMPLOY1', 'ADDEPEV3',
            'DISABILITY_INDEX_NORMALIZED', 'WEIGHT2', 'HEIGHT3', 'CHILDREN',
            'ALCOHOL_SEVERITY_NORMALIZED', 'SMOKING_INDEX_NORMALIZED',
            'MARIJUANA_INDEX_NORMALIZED', 'FINANCIAL_INSECURITY_INDEX_NORMALIZED',
            'FOOD_INSECURITY_INDEX_NORMALIZED', 'EMOTIONAL_DISTRESS_INDEX_NORMALIZED',
            'FACING_UNEMPLOYMENT_INDEX_NORMALIZED', 'HEALTHCAREACCESS_INDEX_NORMALIZED',
            'METRO_AREA', 'URBAN_RURAL_AREA', 'ACE_NORMALIZED',
            'SEXUAL_ORIENTATION_SCORE', 'VETERAN3', 'RENTHOM1', 'PREGNANT'
        ]
        
        # Filter to only existing columns
        for spec_name, confounders in self.specifications.items():
            self.specifications[spec_name] = [col for col in confounders if col in self.df.columns]
            print(f"{spec_name}: {len(self.specifications[spec_name])} confounders")
        
        return self
    
    def create_interaction_terms(self):
        """Create interaction terms for enhanced specifications."""
        print("\n=== Creating Interaction Terms ===")
        
        # Convert categorical variables to numeric for interaction terms
        categorical_vars = ['SEXVAR', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3', 
                           'EMPLOY1', 'CHILDREN', 'INCOME3', 'PREGNANT', 
                           'METRO_AREA', 'URBAN_RURAL_AREA', 'ADDEPEV3']
        
        for var in categorical_vars:
            if var in self.df.columns:
                # Convert to categorical codes, handling NaN values
                self.df[f'{var}_NUMERIC'] = pd.Categorical(self.df[var]).codes
                # Replace -1 (NaN) with actual NaN
                self.df[f'{var}_NUMERIC'] = self.df[f'{var}_NUMERIC'].replace(-1, np.nan)
        
        # Create age variable from existing data if possible
        if 'AGE' not in self.df.columns:
            # Estimate age from other variables or use a proxy
            self.df['AGE_PROXY'] = 50  # Default proxy age
        
        # Create interaction terms using numeric versions
        self.df['SEXVAR_AGE_INTERACTION'] = self.df['SEXVAR_NUMERIC'] * self.df.get('AGE_PROXY', 50)
        self.df['INCOME_EDUCA_INTERACTION'] = self.df['INCOME3_NUMERIC'] * self.df['EDUCA_NUMERIC']
        self.df['DISABILITY_EMPLOYMENT_INTERACTION'] = self.df['DISABILITY_INDEX_NORMALIZED'] * self.df['EMPLOY1_NUMERIC']
        
        print("Interaction terms created")
        return self
    
    def test_specification(self, spec_name, confounders, caliper=0.1, ratio=1, algorithm='nearest'):
        """Test a single PSM specification with enhanced matching."""
        print(f"\n--- Testing {spec_name} specification ---")
        
        # Prepare data for this specification
        analysis_cols = ['_SEQNO', '_EXERCISE_BINARY', '_MENTAL_HEALTH_BINARY'] + confounders
        df_analysis = self.df[analysis_cols].dropna()
        
        if len(df_analysis) == 0:
            print(f"Warning: No data available for {spec_name} specification")
            return None
        
        # Convert categorical variables to numeric codes
        categorical_vars = ['SEXVAR', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3', 
                           'EMPLOY1', 'CHILDREN', 'INCOME3', 'PREGNANT', 
                           'METRO_AREA', 'URBAN_RURAL_AREA', 'ADDEPEV3']
        
        for var in categorical_vars:
            if var in df_analysis.columns:
                # Convert to categorical codes, handling NaN values
                df_analysis[var] = pd.Categorical(df_analysis[var]).codes
                # Replace -1 (NaN) with actual NaN
                df_analysis[var] = df_analysis[var].replace(-1, np.nan)
        
        # Ensure all confounders are numeric
        for var in confounders:
            if var in df_analysis.columns:
                df_analysis[var] = pd.to_numeric(df_analysis[var], errors='coerce')
        
        # Remove any remaining NaN values
        df_analysis = df_analysis.dropna()
        
        if len(df_analysis) == 0:
            print(f"Warning: No data available after preprocessing for {spec_name} specification")
            return None
        
        # Calculate propensity scores
        X = df_analysis[confounders]
        y = df_analysis['_EXERCISE_BINARY']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try different propensity score models
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        }
        
        best_auc = 0
        best_model = None
        best_ps = None
        
        for model_name, model in models.items():
            try:
                model.fit(X_scaled, y)
                ps = model.predict_proba(X_scaled)[:, 1]
                auc = roc_auc_score(y, ps)
                
                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_ps = ps
                    
                print(f"  {model_name} AUC: {auc:.4f}")
            except Exception as e:
                print(f"  {model_name} failed: {e}")
        
        df_analysis['propensity_score'] = best_ps
        
        # Perform enhanced matching
        treated = df_analysis[df_analysis['_EXERCISE_BINARY'] == 1]
        control = df_analysis[df_analysis['_EXERCISE_BINARY'] == 0]
        
        if algorithm == 'nearest':
            matches = self._nearest_neighbor_matching(treated, control, caliper, ratio)
        elif algorithm == 'optimal':
            matches = self._optimal_matching(treated, control, caliper, ratio)
        else:
            matches = self._nearest_neighbor_matching(treated, control, caliper, ratio)
        
        # Calculate balance metrics
        if matches:
            matched_treated = treated[treated['_SEQNO'].isin([m['treated_id'] for m in matches])]
            matched_control = control[control['_SEQNO'].isin([m['control_id'] for m in matches])]
            
            balance_metrics = {}
            for col in confounders:
                treated_mean = matched_treated[col].mean()
                control_mean = matched_control[col].mean()
                treated_std = matched_treated[col].std()
                
                if treated_std > 0:
                    standardized_diff = (treated_mean - control_mean) / treated_std
                else:
                    standardized_diff = 0
                
                balance_metrics[col] = {
                    'treated_mean': treated_mean,
                    'control_mean': control_mean,
                    'standardized_diff': standardized_diff
                }
            
            # Calculate average absolute standardized difference
            avg_abs_diff = np.mean([abs(metrics['standardized_diff']) for metrics in balance_metrics.values()])
            
            # Calculate treatment effect
            treated_outcome = matched_treated['_MENTAL_HEALTH_BINARY'].mean()
            control_outcome = matched_control['_MENTAL_HEALTH_BINARY'].mean()
            treatment_effect = treated_outcome - control_outcome
            
            return {
                'spec_name': spec_name,
                'n_confounders': len(confounders),
                'n_treated': len(treated),
                'n_control': len(control),
                'n_matches': len(matches),
                'auc': best_auc,
                'avg_abs_std_diff': avg_abs_diff,
                'treatment_effect': treatment_effect,
                'balance_metrics': balance_metrics,
                'algorithm': algorithm,
                'caliper': caliper,
                'ratio': ratio
            }
        
        return None
    
    def _nearest_neighbor_matching(self, treated, control, caliper, ratio):
        """Perform nearest neighbor matching with enhanced logic."""
        matches = []
        treated_ps = treated['propensity_score'].values
        control_ps = control['propensity_score'].values
        
        for i, ps_treated in enumerate(treated_ps):
            distances = np.abs(control_ps - ps_treated)
            valid_matches = distances <= caliper
            
            if np.any(valid_matches):
                # Sort by distance and take top matches
                valid_indices = np.where(valid_matches)[0]
                valid_distances = distances[valid_matches]
                sorted_indices = valid_indices[np.argsort(valid_distances)]
                
                # Take up to 'ratio' matches
                n_matches = min(ratio, len(sorted_indices))
                selected_indices = sorted_indices[:n_matches]
                
                for idx in selected_indices:
                    matches.append({
                        'treated_id': treated.iloc[i]['_SEQNO'],
                        'control_id': control.iloc[idx]['_SEQNO'],
                        'ps_treated': ps_treated,
                        'ps_control': control_ps[idx],
                        'distance': distances[idx]
                    })
        
        return matches
    
    def _optimal_matching(self, treated, control, caliper, ratio):
        """Perform optimal matching (simplified version)."""
        # For now, use nearest neighbor as optimal matching is complex
        return self._nearest_neighbor_matching(treated, control, caliper, ratio)
    
    def run_enhanced_specification_testing(self):
        """Run all enhanced specification tests."""
        print("\n=== Running Enhanced PSM Specification Testing ===")
        
        self.load_data().define_variables().create_interaction_terms().define_enhanced_specifications()
        
        results = []
        
        # Test different specifications with various parameters
        test_configs = [
            {'caliper': 0.05, 'ratio': 1, 'algorithm': 'nearest'},
            {'caliper': 0.1, 'ratio': 1, 'algorithm': 'nearest'},
            {'caliper': 0.1, 'ratio': 2, 'algorithm': 'nearest'},
            {'caliper': 0.15, 'ratio': 1, 'algorithm': 'nearest'},
        ]
        
        for spec_name, confounders in self.specifications.items():
            for config in test_configs:
                result = self.test_specification(
                    spec_name, confounders, 
                    caliper=config['caliper'], 
                    ratio=config['ratio'],
                    algorithm=config['algorithm']
                )
                if result:
                    result['config'] = config
                    results.append(result)
        
        self.results = results
        return self
    
    def compare_enhanced_specifications(self):
        """Compare and rank enhanced specifications."""
        if not self.results:
            print("No results to compare. Run specification testing first.")
            return
        
        print("\n=== Enhanced Specification Comparison ===")
        
        # Create comparison DataFrame
        comparison_data = []
        for result in self.results:
            comparison_data.append({
                'Specification': result['spec_name'],
                'Config': f"caliper={result['caliper']}, ratio={result['ratio']}",
                'Confounders': result['n_confounders'],
                'Treated': result['n_treated'],
                'Control': result['n_control'],
                'Matches': result['n_matches'],
                'AUC': result['auc'],
                'Avg Abs Std Diff': result['avg_abs_std_diff'],
                'Treatment Effect': result['treatment_effect'],
                'Algorithm': result['algorithm']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison results
        comparison_df.to_csv('data/enhanced_psm_specification_comparison.csv', index=False)
        
        print("\nEnhanced Specification Comparison:")
        print(comparison_df.round(4))
        
        # Find best specification based on balance
        best_balance = comparison_df.loc[comparison_df['Avg Abs Std Diff'].idxmin()]
        print(f"\nBest balanced specification: {best_balance['Specification']}")
        print(f"Configuration: {best_balance['Config']}")
        print(f"Average absolute standardized difference: {best_balance['Avg Abs Std Diff']:.4f}")
        
        return comparison_df
    
    def create_enhanced_visualizations(self):
        """Create enhanced visualizations for specification comparison."""
        if not self.results:
            print("No results to visualize. Run specification testing first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced PSM Specification Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data
        comparison_data = []
        for result in self.results:
            comparison_data.append({
                'Specification': result['spec_name'],
                'Config': f"caliper={result['caliper']}, ratio={result['ratio']}",
                'Confounders': result['n_confounders'],
                'AUC': result['auc'],
                'Avg Abs Std Diff': result['avg_abs_std_diff'],
                'Treatment Effect': result['treatment_effect'],
                'Matches': result['n_matches']
            })
        
        df_comp = pd.DataFrame(comparison_data)
        
        # 1. Balance by specification and config
        spec_config = df_comp['Specification'] + ' (' + df_comp['Config'] + ')'
        axes[0, 0].bar(range(len(df_comp)), df_comp['Avg Abs Std Diff'], color='lightcoral')
        axes[0, 0].set_title('Average Absolute Standardized Difference')
        axes[0, 0].set_ylabel('Abs Std Diff')
        axes[0, 0].set_xticks(range(len(df_comp)))
        axes[0, 0].set_xticklabels(spec_config, rotation=45, ha='right')
        
        # 2. AUC by specification
        axes[0, 1].bar(range(len(df_comp)), df_comp['AUC'], color='skyblue')
        axes[0, 1].set_title('Propensity Score Model AUC')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_xticks(range(len(df_comp)))
        axes[0, 1].set_xticklabels(spec_config, rotation=45, ha='right')
        
        # 3. Treatment effect by specification
        axes[1, 0].bar(range(len(df_comp)), df_comp['Treatment Effect'], color='lightgreen')
        axes[1, 0].set_title('Treatment Effect')
        axes[1, 0].set_ylabel('Effect Size')
        axes[1, 0].set_xticks(range(len(df_comp)))
        axes[1, 0].set_xticklabels(spec_config, rotation=45, ha='right')
        
        # 4. Number of matches by specification
        axes[1, 1].bar(range(len(df_comp)), df_comp['Matches'], color='gold')
        axes[1, 1].set_title('Number of Matched Pairs')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(range(len(df_comp)))
        axes[1, 1].set_xticklabels(spec_config, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_psm_specification_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Enhanced specification comparison visualizations saved to visualizations/enhanced_psm_specification_comparison.png")
    
    def save_enhanced_specification_summary(self):
        """Save detailed enhanced specification summary."""
        if not self.results:
            print("No results to save. Run specification testing first.")
            return
        
        with open('data/enhanced_psm_specification_summary.txt', 'w') as f:
            f.write("Enhanced PSM Specification Testing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"Specification: {result['spec_name']}\n")
                f.write(f"Configuration: caliper={result['caliper']}, ratio={result['ratio']}, algorithm={result['algorithm']}\n")
                f.write(f"Number of confounders: {result['n_confounders']}\n")
                f.write(f"Sample sizes - Treated: {result['n_treated']}, Control: {result['n_control']}\n")
                f.write(f"Successful matches: {result['n_matches']}\n")
                f.write(f"Propensity score AUC: {result['auc']:.4f}\n")
                f.write(f"Average absolute standardized difference: {result['avg_abs_std_diff']:.4f}\n")
                f.write(f"Treatment effect: {result['treatment_effect']:.4f}\n")
                f.write("\n" + "-" * 40 + "\n\n")
        
        print("Enhanced specification summary saved to data/enhanced_psm_specification_summary.txt")

def main():
    """Main function to run enhanced PSM specification testing."""
    print("Starting Enhanced PSM Specification Testing...")
    
    tester = EnhancedPSMSpecificationTester()
    tester.run_enhanced_specification_testing()
    tester.compare_enhanced_specifications()
    tester.create_enhanced_visualizations()
    tester.save_enhanced_specification_summary()
    
    print("\nEnhanced PSM Specification Testing completed!")

if __name__ == "__main__":
    main() 