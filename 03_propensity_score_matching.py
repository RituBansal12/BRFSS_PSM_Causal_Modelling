#!/usr/bin/env python3
"""
Enhanced Propensity Score Matching for Causal Modeling
Analyzes the impact of exercise on mental health using propensity score matching
to create balanced treatment and control groups.

OPTIMAL SPECIFICATIONS (from enhanced specification testing):
- Model: Random Forest (AUC: 0.7467)
- Caliper: 0.05 (tightest caliper for optimal balance)
- Ratio: 1:1
- Confounders: 25 variables (full_model specification)
- Expected Results: Average SMD = 0.1149, 8,572 matched pairs
- Balance Quality: Significantly improved balance for causal inference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PropensityScoreMatcher:
    """Comprehensive propensity score matching for causal inference."""
    
    def __init__(self, data_path='data/LLCP2023_processed.csv'):
        """Initialize the matcher with data."""
        self.data_path = data_path
        self.df = None
        self.df_matched = None
        self.ps_model = None
        self.treatment_var = None
        self.outcome_var = None
        self.confounders = None
        
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
        
        self.treatment_var = '_EXERCISE_BINARY'
        self.outcome_var = '_MENTAL_HEALTH_BINARY'
        
        print(f"Treatment variable: {self.treatment_var}")
        print(f"Outcome variable: {self.outcome_var}")
        print(f"Number of confounders: {len(self.confounders)}")
        
        return self
    
    def prepare_data(self):
        """Prepare data for propensity score matching."""
        print("\n=== Data Preparation ===")
        
        # Select analysis variables
        analysis_cols = ['_SEQNO', self.treatment_var, self.outcome_var] + self.confounders
        self.df_analysis = self.df[analysis_cols].dropna()
        
        print(f"Analysis dataset: {self.df_analysis.shape[0]} rows, {self.df_analysis.shape[1]} columns")
        
        # Convert categorical variables to numeric codes
        categorical_vars = ['SEXVAR', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3', 
                           'EMPLOY1', 'CHILDREN', 'INCOME3', 'PREGNANT', 
                           'METRO_AREA', 'URBAN_RURAL_AREA', 'ADDEPEV3']
        
        for var in categorical_vars:
            if var in self.df_analysis.columns:
                # Convert to categorical codes, handling NaN values
                self.df_analysis[var] = pd.Categorical(self.df_analysis[var]).codes
                # Replace -1 (NaN) with actual NaN
                self.df_analysis[var] = self.df_analysis[var].replace(-1, np.nan)
        
        # Ensure all confounders are numeric
        for var in self.confounders:
            if var in self.df_analysis.columns:
                self.df_analysis[var] = pd.to_numeric(self.df_analysis[var], errors='coerce')
        
        # Remove any remaining NaN values
        self.df_analysis = self.df_analysis.dropna()
        
        print(f"Final analysis dataset: {self.df_analysis.shape[0]} rows, {self.df_analysis.shape[1]} columns")
        
        # Check treatment distribution
        treatment_counts = self.df_analysis[self.treatment_var].value_counts()
        print(f"Treatment distribution: {treatment_counts.to_dict()}")
        
        # Check outcome distribution
        outcome_counts = self.df_analysis[self.outcome_var].value_counts()
        print(f"Outcome distribution: {outcome_counts.to_dict()}")
        
        return self
    
    def calculate_propensity_scores(self):
        """Calculate propensity scores using Random Forest (optimal model from specification testing)."""
        print("\n=== Calculating Propensity Scores ===")
        
        X = self.df_analysis[self.confounders]
        y = self.df_analysis[self.treatment_var]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use Random Forest (optimal model from specification testing)
        self.ps_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=5,
            n_jobs=-1
        )
        self.ps_model.fit(X_scaled, y)
        
        # Calculate propensity scores
        self.df_analysis['propensity_score'] = self.ps_model.predict_proba(X_scaled)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y, self.df_analysis['propensity_score'])
        print(f"Random Forest propensity score model AUC: {auc:.4f}")
        
        # Print model summary
        print("\nPropensity Score Model Summary:")
        print(f"Model: Random Forest (n_estimators=100, max_depth=5)")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'confounder': self.confounders,
            'importance': self.ps_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important confounders:")
        print(feature_importance.head(10))
        
        return self
    
    def assess_balance_before_matching(self):
        """Assess covariate balance before matching."""
        print("\n=== Assessing Balance Before Matching ===")
        
        treated = self.df_analysis[self.df_analysis[self.treatment_var] == 1]
        control = self.df_analysis[self.df_analysis[self.treatment_var] == 0]
        
        balance_data = []
        
        for col in self.confounders:
            treated_mean = treated[col].mean()
            control_mean = control[col].mean()
            treated_std = treated[col].std()
            
            if treated_std > 0:
                smd = (treated_mean - control_mean) / treated_std
            else:
                smd = 0
            
            balance_data.append({
                'variable': col,
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'smd': smd
            })
        
        self.balance_before = pd.DataFrame(balance_data)
        
        # Calculate summary statistics
        avg_abs_smd = abs(self.balance_before['smd']).mean()
        max_abs_smd = abs(self.balance_before['smd']).max()
        poor_balance_count = (abs(self.balance_before['smd']) > 0.1).sum()
        
        print(f"Average absolute SMD: {avg_abs_smd:.4f}")
        print(f"Maximum absolute SMD: {max_abs_smd:.4f}")
        print(f"Variables with poor balance (|SMD| > 0.1): {poor_balance_count}")
        
        # Save balance results
        self.balance_before.to_csv('data/balance_before_matching.csv', index=False)
        print("Balance before matching saved to: data/balance_before_matching.csv")
        
        return self
    
    def perform_matching(self, caliper=0.05, ratio=1):
        """Perform propensity score matching with optimal specifications."""
        print(f"\n=== Performing Matching (caliper={caliper}, ratio={ratio}) ===")
        print("Using optimal full_model specification with tightest caliper for best balance")
        
        treated = self.df_analysis[self.df_analysis[self.treatment_var] == 1]
        control = self.df_analysis[self.df_analysis[self.treatment_var] == 0]
        
        # Simple nearest neighbor matching with caliper
        matches = []
        treated_ps = treated['propensity_score'].values
        control_ps = control['propensity_score'].values
        
        for i, ps_treated in enumerate(treated_ps):
            distances = np.abs(control_ps - ps_treated)
            valid_matches = distances <= caliper
            
            if np.any(valid_matches):
                best_match = np.argmin(distances[valid_matches])
                control_indices = np.where(valid_matches)[0]
                matched_control_idx = control_indices[best_match]
                
                matches.append({
                    'treated_id': treated.iloc[i]['_SEQNO'],
                    'control_id': control.iloc[matched_control_idx]['_SEQNO'],
                    'ps_treated': ps_treated,
                    'ps_control': control_ps[matched_control_idx],
                    'distance': distances[valid_matches][best_match]
                })
        
        if matches:
            # Create matched dataset
            matched_ids = []
            for match in matches:
                matched_ids.extend([match['treated_id'], match['control_id']])
            
            self.df_matched = self.df_analysis[self.df_analysis['_SEQNO'].isin(matched_ids)].copy()
            
            # Save matching pairs
            self.matched_pairs = pd.DataFrame(matches)
            
            print(f"Successful matches: {len(matches)}")
            print(f"Matched dataset: {len(self.df_matched)} rows")
        else:
            print("No matches found with current caliper")
            self.df_matched = None
            self.matched_pairs = None
        
        return self
    
    def assess_balance_after_matching(self):
        """Assess covariate balance after matching."""
        if self.df_matched is None:
            print("No matched dataset available. Run perform_matching first.")
            return self
        
        print("\n=== Assessing Balance After Matching ===")
        
        matched_treated = self.df_matched[self.df_matched[self.treatment_var] == 1]
        matched_control = self.df_matched[self.df_matched[self.treatment_var] == 0]
        
        balance_data = []
        
        for col in self.confounders:
            treated_mean = matched_treated[col].mean()
            control_mean = matched_control[col].mean()
            treated_std = matched_treated[col].std()
            
            if treated_std > 0:
                smd = (treated_mean - control_mean) / treated_std
            else:
                smd = 0
            
            balance_data.append({
                'variable': col,
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'smd': smd
            })
        
        self.balance_after = pd.DataFrame(balance_data)
        
        # Calculate summary statistics
        avg_abs_smd = abs(self.balance_after['smd']).mean()
        max_abs_smd = abs(self.balance_after['smd']).max()
        poor_balance_count = (abs(self.balance_after['smd']) > 0.1).sum()
        
        print(f"Average absolute SMD: {avg_abs_smd:.4f}")
        print(f"Maximum absolute SMD: {max_abs_smd:.4f}")
        print(f"Variables with poor balance (|SMD| > 0.1): {poor_balance_count}")
        
        # Save balance results
        self.balance_after.to_csv('data/balance_after_matching.csv', index=False)
        print("Balance after matching saved to: data/balance_after_matching.csv")
        
        return self
    
    def calculate_treatment_effect(self):
        """Calculate the average treatment effect."""
        if self.df_matched is None:
            print("No matched dataset available. Run perform_matching first.")
            return None
        
        print("\n=== Calculating Treatment Effect ===")
        
        matched_treated = self.df_matched[self.df_matched[self.treatment_var] == 1]
        matched_control = self.df_matched[self.df_matched[self.treatment_var] == 0]
        
        # Calculate treatment effect
        treated_outcome = matched_treated[self.outcome_var].mean()
        control_outcome = matched_control[self.outcome_var].mean()
        ate = treated_outcome - control_outcome
        
        # Calculate standard error (simple approach)
        treated_se = matched_treated[self.outcome_var].std() / np.sqrt(len(matched_treated))
        control_se = matched_control[self.outcome_var].std() / np.sqrt(len(matched_control))
        ate_se = np.sqrt(treated_se**2 + control_se**2)
        
        # Calculate confidence interval (95%)
        ate_ci_lower = ate - 1.96 * ate_se
        ate_ci_upper = ate + 1.96 * ate_se
        
        results = pd.DataFrame({
            'Metric': ['Treated Outcome', 'Control Outcome', 'Treatment Effect', 'SE', 'CI Lower', 'CI Upper'],
            'Value': [treated_outcome, control_outcome, ate, ate_se, ate_ci_lower, ate_ci_upper]
        })
        
        print("Treatment Effect Results:")
        print(results)
        
        # Save results
        results.to_csv('data/treatment_effect_results.csv', index=False)
        print("Treatment effect results saved to: data/treatment_effect_results.csv")
        
        self.treatment_effect_results = results
        return self
    
    def create_visualizations(self):
        """Create visualizations for the PSM analysis."""
        if self.df_matched is None:
            print("No matched dataset available. Run perform_matching first.")
            return self
        
        print("\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Propensity Score Matching Analysis: Exercise Impact on Mental Health', 
                    fontsize=16, fontweight='bold')
        
        # 1. Propensity score distribution
        ax1 = axes[0, 0]
        treated_ps = self.df_analysis[self.df_analysis[self.treatment_var] == 1]['propensity_score']
        control_ps = self.df_analysis[self.df_analysis[self.treatment_var] == 0]['propensity_score']
        
        ax1.hist(control_ps, bins=30, alpha=0.7, label='Control', color='blue')
        ax1.hist(treated_ps, bins=30, alpha=0.7, label='Treated', color='red')
        ax1.set_xlabel('Propensity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Propensity Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Balance before matching
        ax2 = axes[0, 1]
        smd_before = self.balance_before['smd']
        ax2.barh(range(len(smd_before)), smd_before, color='red', alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Standardized Mean Difference')
        ax2.set_ylabel('Variables')
        ax2.set_title('Balance Before Matching')
        ax2.grid(True, alpha=0.3)
        
        # 3. Balance after matching
        ax3 = axes[0, 2]
        smd_after = self.balance_after['smd']
        colors = ['green' if abs(x) <= 0.1 else 'red' for x in smd_after]
        ax3.barh(range(len(smd_after)), smd_after, color=colors, alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax3.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Standardized Mean Difference')
        ax3.set_ylabel('Variables')
        ax3.set_title('Balance After Matching')
        ax3.grid(True, alpha=0.3)
        
        # 4. Balance improvement
        ax4 = axes[1, 0]
        improvement = abs(smd_before) - abs(smd_after)
        colors = ['green' if x > 0 else 'red' for x in improvement]
        ax4.barh(range(len(improvement)), improvement, color=colors, alpha=0.7)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Balance Improvement (|SMD Before| - |SMD After|)')
        ax4.set_ylabel('Variables')
        ax4.set_title('Balance Improvement')
        ax4.grid(True, alpha=0.3)
        
        # 5. Treatment effect
        ax5 = axes[1, 1]
        matched_treated = self.df_matched[self.df_matched[self.treatment_var] == 1]
        matched_control = self.df_matched[self.df_matched[self.treatment_var] == 0]
        
        treated_outcome = matched_treated[self.outcome_var].mean()
        control_outcome = matched_control[self.outcome_var].mean()
        
        bars = ax5.bar(['Control', 'Treated'], [control_outcome, treated_outcome], 
                      color=['blue', 'red'], alpha=0.7)
        ax5.set_ylabel('Mental Health Score')
        ax5.set_title('Treatment Effect')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, [control_outcome, treated_outcome]):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Sample size comparison
        ax6 = axes[1, 2]
        original_treated = len(self.df_analysis[self.df_analysis[self.treatment_var] == 1])
        original_control = len(self.df_analysis[self.df_analysis[self.treatment_var] == 0])
        matched_treated_n = len(matched_treated)
        matched_control_n = len(matched_control)
        
        x = np.arange(2)
        width = 0.35
        
        ax6.bar(x - width/2, [original_treated, original_control], width, 
               label='Before Matching', alpha=0.7)
        ax6.bar(x + width/2, [matched_treated_n, matched_control_n], width, 
               label='After Matching', alpha=0.7)
        
        ax6.set_xlabel('Group')
        ax6.set_ylabel('Sample Size')
        ax6.set_title('Sample Size Comparison')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Treated', 'Control'])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/psm_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to: visualizations/psm_analysis_results.png")
        return self
    
    def save_results(self):
        """Save all results from the PSM analysis."""
        print("\n=== Saving Results ===")
        
        # Save matched dataset
        if self.df_matched is not None:
            self.df_matched.to_csv('data/best_matched_llcp2023.csv', index=False)
            print("Matched dataset saved to: data/best_matched_llcp2023.csv")
            print("Note: Using optimal full_model specification (caliper=0.05, Random Forest)")
        
        # Save matching pairs
        if hasattr(self, 'matched_pairs') and len(self.matched_pairs) > 0:
            self.matched_pairs.to_csv('data/matching_pairs.csv', index=False)
            print("Matching pairs saved to: data/matching_pairs.csv")
        
        return self
    
    def run_full_analysis(self, caliper=0.05, ratio=1):
        """Run the complete propensity score matching analysis with optimal specifications."""
        print("=" * 60)
        print("ENHANCED PROPENSITY SCORE MATCHING ANALYSIS")
        print("Exercise Impact on Mental Health")
        print("Using Optimal Specifications: caliper=0.05, ratio=1:1, Random Forest")
        print("Expected Results: Average SMD = 0.1149, 8,572 matched pairs")
        print("=" * 60)
        
        (self.load_data()
         .define_variables()
         .prepare_data()
         .calculate_propensity_scores()
         .assess_balance_before_matching()
         .perform_matching(caliper=caliper, ratio=ratio)
         .assess_balance_after_matching()
         .calculate_treatment_effect()
         .create_visualizations()
         .save_results())
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED")
        print("=" * 60)
        
        return self

def main():
    """Main function to run enhanced PSM analysis with optimal specifications."""
    print("Starting Enhanced Propensity Score Matching Analysis...")
    print("Using optimal specifications from enhanced specification testing:")
    print("- Model: Random Forest (AUC: 0.7467)")
    print("- Caliper: 0.05 (tightest caliper for optimal balance)")
    print("- Ratio: 1:1")
    print("- Confounders: 25 variables (full_model specification)")
    print("- Expected balance: Average SMD = 0.1149")
    
    matcher = PropensityScoreMatcher()
    matcher.run_full_analysis(caliper=0.05, ratio=1)
    
    print("\nEnhanced PSM Analysis completed!")

if __name__ == "__main__":
    main() 