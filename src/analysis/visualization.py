import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    def __init__(self, data):
        self.data = data
    
    # Function for Univariate Analysis of Numerical Variables
    def plot_numerical_distribution(self, numerical_cols=None):
        """Plot histograms for numerical columns"""
        if numerical_cols is None:
            numerical_cols = self.data.select_dtypes(include=[np.int64, np.float64]).columns.tolist()
        
        if not numerical_cols:
            print("No numerical columns found to plot")
            return
            
        n_cols = min(2, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) 
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, col in enumerate(numerical_cols):
            self.data[col].hist(bins=20, color='green', edgecolor='black', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=10)
        
        # Hide any unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle('Distribution of Numerical Features', y=1.02)
        plt.tight_layout()
        plt.show()

    # Function for Univariate Analysis of Categorical Variables
    def plot_categorical_distribution(self, categorical_cols=None):
        """Plot count plots for categorical columns"""
        if categorical_cols is None:
            categorical_cols = self.data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if not categorical_cols:
            print("No categorical columns found to plot")
            return
            
        n_cols = 3
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, col in enumerate(categorical_cols):
            sns.countplot(data=self.data, x=col, palette='Set2', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=10)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide any unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle('Distribution of Categorical Features', y=1.02)
        plt.tight_layout()
        plt.show()

    # Combined univariate analysis
    def plot_univariate_analysis(self, columns=None):
        """Automatically detect and plot appropriate visualizations for all columns"""
        if columns is None:
            columns = self.data.columns.tolist()
        
        numerical_cols = self.data[columns].select_dtypes(include=[np.int64, np.float64]).columns.tolist()
        categorical_cols = self.data[columns].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if numerical_cols:
            self.plot_numerical_distribution(numerical_cols)
        if categorical_cols:
            self.plot_categorical_distribution(categorical_cols)
    # Function for analysis of Loss Ratios varies from by columns
    def plot_loss_ratios(self, col=None):
        if col in self.data.columns:
            loss_ratio_by_col = self.data.groupby(col).agg(
                TotalPremium=('TotalPremium', 'sum'),
                TotalClaims=('TotalClaims', 'sum')
            ).reset_index()
            loss_ratio_by_col['LossRatio'] = loss_ratio_by_col['TotalClaims'] / loss_ratio_by_col['TotalPremium']
            print(f"\n--- Loss Ratio by {col} ---")
            print(loss_ratio_by_col.sort_values('LossRatio', ascending=False))
            # Plotting the loss ratio by column
            plt.figure(figsize=(12, 7))
            sns.barplot(x='LossRatio', y=col, data=loss_ratio_by_col.sort_values('LossRatio', ascending=False), palette='coolwarm')
            plt.title(f'Loss Ratio by {col}')
            plt.xlabel('Loss Ratio (Total Claims / Total Premium)')
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()
    def correlation_heatmap(self, corr_matrix=None):
        """Plot a heatmap of the correlation matrix for numerical features"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Key Numerical Features')
        plt.tight_layout()
        plt.show()

    def plot_claim_frequency(self, monthly_data):
        plt.figure(figsize=(15, 6))
        sns.lineplot(x='TransactionMonth', y='ClaimFrequency', data=monthly_data, marker='o')
        plt.title('Monthly Claim Frequency Over Time')
        plt.xlabel('Transaction Month')
        plt.ylabel('Claim Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def plot_claim_severity(self, monthly_data):
        plt.figure(figsize=(15, 6))
        sns.lineplot(x='TransactionMonth', y='ClaimSeverity', data=monthly_data, marker='o')
        plt.title('Monthly Claim Severity Over Time')
        plt.xlabel('Transaction Month')
        plt.ylabel('Claim Severity (Total Claims Amount / Policies With Claims)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def plot_loss_ratio(self, monthly_data):
        plt.figure(figsize=(15, 6))
        sns.lineplot(x='TransactionMonth', y='MonthlyLossRatio', data=monthly_data, marker='o')
        plt.title('Monthly Loss Ratio Over Time')
        plt.xlabel('Transaction Month')
        plt.ylabel('Monthly Loss Ratio (Total Claims Amount / Total Premium Amount)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_monthly_trends(self):
        if 'TransactionMonth' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['TransactionMonth']):
            # Aggregate monthly data
            monthly_data = self.data.groupby(self.data['TransactionMonth'].dt.to_period('M')).agg(
                TotalPolicies=('PolicyID', 'nunique'),
                PoliciesWithClaims=('TotalClaims', lambda x: (x > 0).sum()),
                TotalClaimsAmount=('TotalClaims', 'sum'),
                TotalPremiumAmount=('TotalPremium', 'sum')
            ).reset_index()
            
            # Calculate metrics
            monthly_data['ClaimFrequency'] = monthly_data['PoliciesWithClaims'] / monthly_data['TotalPolicies']
            monthly_data['ClaimSeverity'] = monthly_data['TotalClaimsAmount'] / monthly_data['PoliciesWithClaims']
            monthly_data['MonthlyLossRatio'] = monthly_data['TotalClaimsAmount'] / monthly_data['TotalPremiumAmount']
            monthly_data['TransactionMonth'] = monthly_data['TransactionMonth'].astype(str)  # Convert Period to string

            # Print results
            print("\n--- Monthly Trends ---")
            print(monthly_data)

            # Plot Claim Frequency
            self.plot_claim_frequency(monthly_data)

            # Plot Claim Severity
            self.plot_claim_severity(monthly_data)

            # Plot Loss Ratio
            self.plot_loss_ratio(monthly_data)
            
            return monthly_data
            
        else:
            print("\n--- Skipping Temporal Trend Analysis: 'TransactionMonth' is not in datetime format or not found. ---")
            return None

  