# import libraries
import matplotlib as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns # For enhanced visualizations

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer

# Set plot style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# --- Your Model Class Definition ---
class Model:
    def __init__(self, df):
        self.df = df 
        self.lr = None
        self.dt = None
        self.rfr = None
        self.xgb = None
        self.label_encoders = {}
    def data_encode(self):
        self.df = self.df.drop_duplicates(keep="first")
        self.df['make'] = self.df['make'].str.strip()
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = self.df.select_dtypes(include=['number']).columns
        # Label encoding
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        return self.df
    def split_data(self, X, y, test_size=0.2, random_state=42):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        """
        Initializes and trains various regression models.
        """
        print("\nTraining models...")
        # Initialize the models
        lr_model = LinearRegression()
        dt_model = DecisionTreeRegressor(random_state=42)
        rfr_model = RandomForestRegressor(random_state=42)
        xgb_model = xgb.XGBRegressor(random_state=42)
        
        # Train the models
        print("Fitting models...")
        print(f"  Linear Regression...")
        self.lr = lr_model.fit(X_train, y_train)
        print(f"  Decision Tree...")
        self.dt = dt_model.fit(X_train, y_train)
        print(f"  Random Forest...")
        self.rfr = rfr_model.fit(X_train, y_train)
        print(f"  XGBoost...")
        self.xgb = xgb_model.fit(X_train, y_train)
        print("Models trained successfully.")
        return self.lr, self.dt, self.rfr, self.xgb

    def evaluate_models(self, model, X_test, y_test):
        """
        Evaluates the trained models using MAE, MSE, and R2.
        """
        print("\nEvaluating models...")
        # Make predictions
        y_pred = model.predict(X_test)
        # accuracy
        print(f"  Evaluating {model.__class__.__name__}...")
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # print accuracy
        print(f"  Accuracy: {model.score(X_test, y_test):.4f}")
        return mae, mse, r2, y_pred

    def visualize_results(self, models, mae_scores, mse_scores, r2_scores):
        """
        Visualizes actual vs. predicted values for each model.
        """
        print("\nGenerating visualizations...")
        for model in models:
            # Plot MAE scores
            plt.figure(figsize=(6, 4))
            plt.bar(model, mae_scores, color='skyblue')
            plt.xlabel('Models')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.title('Comparison of MAE Scores')
            plt.xticks(rotation=45)
            plt.show()

            # Plot MSE scores
            plt.figure(figsize=(6, 4))
            plt.bar(models, mse_scores, color='lightgreen')
            plt.xlabel('Models')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.title('Comparison of MSE Scores')
            plt.xticks(rotation=45)
            plt.show()

            # Plot R-squared scores
            plt.figure(figsize=(6, 4))
            plt.bar(models, r2_scores, color='salmon')
            plt.xlabel('Models')
            plt.ylabel('R-squared Score')
            plt.title('Comparison of R-squared Scores')
            plt.xticks(rotation=45)
            plt.show()