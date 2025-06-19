# import libraries
import os
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = None

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found in {self.data_dir}")
        if filename.endswith('.txt'):
            self.data = pd.read_csv(file_path, sep="|")
        else:
            self.data = pd.read_csv(file_path)
        #convert 'TransactionMonth' column to datetime if it exists
        if 'TransactionMonth' in self.data.columns:
            self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'], errors='coerce')
        #convert 'VehicleIntroDate' column to datetime if it exists
        if 'VehicleIntroDate' in self.data.columns:
            self.data['VehicleIntroDate'] = pd.to_datetime(self.data['VehicleIntroDate'], errors='coerce')
        return self.data