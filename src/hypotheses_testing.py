# import the libraries
import numpy as np
import pandas as pd
from scipy import stats # For ttest_ind, f_oneway, kruskal, chi2_contingency, mannwhitneyu

class HypothesisTesting:
    def __init__(self, data):
        self.data = data.copy() # Work on a copy to avoid modifying original df

        # Ensure necessary KPIs are derived/cleaned before testing
        # Handle division by zero for LossRatio
        self.data['LossRatio'] = self.data.apply(
            lambda row: row['TotalClaims'] / row['TotalPremium'] if row['TotalPremium'] > 0 else np.nan,
            axis=1
        )
        # Create HasClaim binary column
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        # Calculate NetPremium for profit/margin proxy
        self.data['NetPremium'] = self.data['TotalPremium'] - self.data['TotalClaims']
        

    def _get_groups_data(self, group_col, kpi_col, dropna=True):
        """Helper to get data grouped by a column for statistical tests."""
        grouped = self.data.groupby(group_col)[kpi_col]
        # Filter out groups with insufficient data (e.g., less than 2 samples)
        # and drop NA if specified
        return [group.dropna() if dropna else group for _, group in grouped if len(group.dropna()) > 1]

    def test_province_risk(self, alpha=0.05):
        print("\n--- Hypothesis: No risk differences across Provinces ---")
        results = {}

        # 1. Test Loss Ratio (Numerical)
        province_loss_ratios = self._get_groups_data('Province', 'LossRatio')
        
        if len(province_loss_ratios) > 1: # Ensure more than one province with data
            # Use Kruskal-Wallis as LossRatio is often not normally distributed
            kruskal_stat, p_value_kruskal = stats.kruskal(*province_loss_ratios)
            results['LossRatio_Kruskal'] = {'statistic': kruskal_stat, 'p_value': p_value_kruskal}
            print(f"Loss Ratio (Kruskal-Wallis): Stat={kruskal_stat:.4f}, P={p_value_kruskal:.4f}")

            if p_value_kruskal < alpha:
                print(f"  --> Reject H₀ for Loss Ratio. Significant differences exist across provinces (p={p_value_kruskal:.4f}).")
                print("  --> (Further post-hoc analysis needed to identify specific differing provinces).")
            else:
                print(f"  --> Fail to reject H₀ for Loss Ratio. No significant differences across provinces (p={p_value_kruskal:.4f}).")
        else:
            print("  --> Not enough data to test Loss Ratio across provinces.")

        # 2. Test Claim Frequency (Categorical)
        contingency_table = pd.crosstab(self.data['Province'], self.data['HasClaim'])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1: # Ensure enough data for test
            chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
            results['ClaimFrequency_Chi2'] = {'statistic': chi2_stat, 'p_value': p_value_chi2}
            print(f"Claim Frequency (Chi-squared): Stat={chi2_stat:.4f}, P={p_value_chi2:.4f}")

            if p_value_chi2 < alpha:
                print(f"  --> Reject H₀ for Claim Frequency. Claim frequency is dependent on province (p={p_value_chi2:.4f}).")
            else:
                print(f"  --> Fail to reject H₀ for Claim Frequency. Claim frequency is independent of province (p={p_value_chi2:.4f}).")
        else:
            print("  --> Not enough data to test Claim Frequency across provinces.")

        return results
    def test_zipcode_risk(self, alpha=0.05):
        print("\n--- Hypothesis: No risk differences across Zip Codes ---")
        results = {}

        # 1. Test Loss Ratio (Numerical)
        zipcode_loss_ratios = self._get_groups_data('PostalCode', 'LossRatio')
        
        if len(zipcode_loss_ratios) > 1:
            # Use Kruskal-Wallis as LossRatio is often not normally distributed
            kruskal_stat, p_value_kruskal = stats.kruskal(*zipcode_loss_ratios)
            results['LossRatio_Kruskal'] = {'statistic': kruskal_stat, 'p_value': p_value_kruskal}
            print(f"Loss Ratio (Kruskal-Wallis): Stat={kruskal_stat:.4f}, P={p_value_kruskal:.4f}")

            if p_value_kruskal < alpha:
                print(f"  --> Reject H₀ for Loss Ratio. Significant differences exist across zip codes (p={p_value_kruskal:.4f}).")
                print("  --> (Further post-hoc analysis needed to identify specific differing zip codes).")
            else:
                print(f"  --> Fail to reject H₀ for Loss Ratio. No significant differences across zip codes (p={p_value_kruskal:.4f}).")
        else:
            print("  --> Not enough data to test Loss Ratio across zip codes.")
        # 2. Test Claim Frequency (Categorical)
        contingency_table = pd.crosstab(self.data['PostalCode'], self.data['HasClaim'])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
            results['ClaimFrequency_Chi2'] = {'statistic': chi2_stat, 'p_value': p_value_chi2}
            print(f"Claim Frequency (Chi-squared): Stat={chi2_stat:.4f}, P={p_value_chi2:.4f}")

            if p_value_chi2 < alpha:
                print(f"  --> Reject H₀ for Claim Frequency. Claim frequency is dependent on zip code (p={p_value_chi2:.4f}).")
            else:
                print(f"  --> Fail to reject H₀ for Claim Frequency. Claim frequency is independent of zip code (p={p_value_chi2:.4f}).")
        return results
    # no significant margin (profit) difference between zip codes 
    def test_zipcode_margin(self, alpha=0.05):
        print("\n--- Hypothesis: No significant margin differences across Zip Codes ---")
        results = {}

        # 1. Test Net Premium (Numerical)
        zipcode_net_premiums = self._get_groups_data('PostalCode', 'NetPremium')
        
        if len(zipcode_net_premiums) > 1:
            # Use Kruskal-Wallis as NetPremium is often not normally distributed
            kruskal_stat, p_value_kruskal = stats.kruskal(*zipcode_net_premiums)
            results['NetPremium_Kruskal'] = {'statistic': kruskal_stat, 'p_value': p_value_kruskal}
            print(f"Net Premium (Kruskal-Wallis): Stat={kruskal_stat:.4f}, P={p_value_kruskal:.4f}")

            if p_value_kruskal < alpha:
                print(f"  --> Reject H₀ for Net Premium. Significant differences exist across zip codes (p={p_value_kruskal:.4f}).")
                print("  --> (Further post-hoc analysis needed to identify specific differing zip codes).")
            else:
                print(f"  --> Fail to reject H₀ for Net Premium. No significant differences across zip codes (p={p_value_kruskal:.4f}).")
        else:
            print("  --> Not enough data to test Net Premium across zip codes.")
            
        return results
    def test_gender_risk(self, alpha=0.05):
        print("\n--- Hypothesis: No significant risk differences between Women and Men ---")
        results = {}
        
        # Filter out 'Not specified' Gender for this specific test
        gender_data = self.data[self.data['Gender'].isin(['Male', 'Female'])].copy()

        # 1. Test Loss Ratio (Numerical)
        male_loss_ratios = gender_data[gender_data['Gender'] == 'Male']['LossRatio'].dropna()
        female_loss_ratios = gender_data[gender_data['Gender'] == 'Female']['LossRatio'].dropna()

        if len(male_loss_ratios) > 1 and len(female_loss_ratios) > 1:
            # Mann-Whitney U test (non-parametric alternative to t-test, generally safer for non-normal financial data)
            stat_mw, p_value_mw = stats.mannwhitneyu(male_loss_ratios, female_loss_ratios, alternative='two-sided')
            results['LossRatio_MannWhitneyU'] = {'statistic': stat_mw, 'p_value': p_value_mw}
            print(f"Loss Ratio (Mann-Whitney U) for Male vs Female: Stat={stat_mw:.4f}, P={p_value_mw:.4f}")

            if p_value_mw < alpha:
                print(f"  --> Reject H₀ for Loss Ratio. Significant differences exist between Male and Female (p={p_value_mw:.4f}).")
            else:
                print(f"  --> Fail to reject H₀ for Loss Ratio. No significant differences between Male and Female (p={p_value_mw:.4f}).")
        else:
            print("  --> Not enough data to test Loss Ratio between genders.")

        # 2. Test Claim Frequency (Categorical)
        contingency_table = pd.crosstab(gender_data['Gender'], gender_data['HasClaim'])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
            results['ClaimFrequency_Chi2'] = {'statistic': chi2_stat, 'p_value': p_value_chi2}
            print(f"Claim Frequency (Chi-squared) for Gender: Stat={chi2_stat:.4f}, P={p_value_chi2:.4f}")

            if p_value_chi2 < alpha:
                print(f"  --> Reject H₀ for Claim Frequency. Claim frequency is dependent on Gender (p={p_value_chi2:.4f}).")
            else:
                print(f"  --> Fail to reject H₀ for Claim Frequency. Claim frequency is independent of Gender (p={p_value_chi2:.4f}).")
        else:
            print("  --> Not enough data to test Claim Frequency between genders.")
            
        return results

