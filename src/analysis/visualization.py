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
            self.data[col].hist(bins=20, color='skyblue', edgecolor='black', ax=axes[i])
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