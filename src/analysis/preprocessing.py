import pandas as pd
import numpy as np
class Preprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fill_missing_values(self):
        for column in self.df.select_dtypes(include=[np.number]).columns:
            self.df[column].fillna(self.df[column].mean(), inplace=True)
        for column in self.df.select_dtypes(include=[object]).columns:
            self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        return self.df

    def drop_missing_columns(self, threshold=0.05):
        # Drop columns with more than 5% missing values
        missing_percent = self.df.isnull().mean()
        cols_to_drop = missing_percent[missing_percent > threshold].index
        print(f"Dropping columns with > {threshold*100}% missing values:\n{list(cols_to_drop)}\n")
        return self.df.drop(columns=cols_to_drop)

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self.df
    def remove_outliers(self, z_threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            self.df = self.df[z_scores < z_threshold]
        return self.df
# do all preprocessing steps
    def preprocess(self):
        self.df = self.drop_missing_columns()
        self.df = self.fill_missing_values()
        self.df = self.remove_duplicates()
        self.df = self.remove_outliers()
        return self.df
    
    