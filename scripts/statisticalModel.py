import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

class InsuranceModeling:
    def __init__(self, df):
        """
        Initialize with the dataframe.
        """
        self.data = df

    def wrangle_data(self, null_threshold=0.5, low_cardinality_threshold=3, high_cardinality_threshold=1000000):
        """
        This function performs the following:
        - Drops columns with a high percentage of null values.
        - Drops categorical columns with low and high cardinality.
        - Filters for outliers in numeric columns using the IQR method.
        - Fills missing numeric values with the mean.
        - Fills missing categorical values with the mode.
        """
        
        # 1. Drop columns with a high percentage of missing values
        missing_percentage = self.data.isnull().mean()
        cols_to_drop = missing_percentage[missing_percentage > null_threshold].index
        self.data = self.data.drop(columns=cols_to_drop)

        # Drop unnecessary columns
        irrelevantColumns=['UnderwrittenCoverID', 'PolicyID',"MainCrestaZone","SubCrestaZone",
                           "bodytype","ExcessSelected","CoverCategory","CoverType","mmcode"]
        self.data = self.data.drop(columns=irrelevantColumns)
        
        # Replace comma with period and then convert to float
        self.data['CapitalOutstanding'] = self.data['CapitalOutstanding'].str.replace(',', '.').astype(float)

        
        # 2. Drop low and high cardinality columns
        cardinality = self.data.select_dtypes(include="object").nunique()
        low_cardinality_cols = cardinality[cardinality < low_cardinality_threshold].index
        high_cardinality_cols = cardinality[cardinality > high_cardinality_threshold].index
        self.data = self.data.drop(columns=list(low_cardinality_cols) + list(high_cardinality_cols))
        
        # 3. Filter for outliers in numeric columns using the IQR method
        for col in self.data.select_dtypes(include=['int64', 'float64']).columns:
            Q1 = self.data[col].quantile(0.01)
            Q3 = self.data[col].quantile(0.99)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            self.data = self.data[(self.data[col] >= Q1 - outlier_step) & (self.data[col] <= Q3 + outlier_step)]
        
        # 4. Fill missing numeric values with the mean
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # 5. Fill missing categorical values with the mode
        categorical_cols = self.data.select_dtypes(include="object").columns
        self.data[categorical_cols] = self.data[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col)
        
        return self.data
        
    def feature_engineering(self):
        # Convert 'TransactionMonth' to datetime if not already
        self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'], errors='coerce')
        
        # Create new features
        self.data['VehicleAge'] = pd.to_datetime('today').year - self.data['RegistrationYear']
        self.data['PremiumPerUnitSumInsured'] = self.data['TotalPremium'] / (self.data['SumInsured'] + 1e-5)
        self.data['ClaimToPremiumRatio'] = self.data['TotalClaims'] / (self.data['TotalPremium'] + 1e-5)
        self.data['VehiclePowerToWeightRatio'] = self.data['kilowatts'] / (self.data['cubiccapacity'] + 1e-5)

        # Extract features from 'TransactionMonth'
        self.data['TransactionYear'] = self.data['TransactionMonth'].dt.year
        self.data['TransactionMonthName'] = self.data['TransactionMonth'].dt.month_name()
        self.data['TransactionQuarter'] = self.data['TransactionMonth'].dt.quarter


        return self.data

