import pandas as pd
import numpy as np
# Descriptive Statistics: Calculate the variability for numerical features such as TotalPremium, TotalClaim, etc.
class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def descriptive_statistics(self):
        # Descriptive statistics for numerical features
        descriptive_stats = self.df.describe()
        return descriptive_stats

    def data_types(self):
        # Data types of each column
        data_types = self.df.dtypes
        return data_types
    # missing value analaysis
    def missing_values(self):
        # Percentage of missing values in each column
        missing_values = self.df.isnull().mean() * 100
        return missing_values[missing_values > 0]
    def outlier_analysis(self):
        # Outlier analysis using Z-score
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            outliers[col] = np.sum(z_scores > 3)
        return outliers
    def unique_values(self):
        unique_values = {col: self.df[col].nunique() for col in self.df.columns}
        return unique_values
    