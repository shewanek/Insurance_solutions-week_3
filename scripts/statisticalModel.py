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
        
        # Create new features
        self.data['VehicleAge'] = 2015 - self.data['RegistrationYear']
        self.data['PremiumPerUnitSumInsured'] = self.data['TotalPremium'] / (self.data['SumInsured'] + 1e-5)
        self.data['ClaimToPremiumRatio'] = self.data['TotalClaims'] / (self.data['TotalPremium'] + 1e-5)
        self.data['VehiclePowerToWeightRatio'] = self.data['kilowatts'] / (self.data['cubiccapacity'] + 1e-5)

        

        return self.data

    def encode_categorical_data(self):
        """
        Convert categorical data into a numeric format using one-hot encoding.
        """
        self.data = pd.get_dummies(self.data, drop_first=True)
        return self.data
        
    def train_test_split(self):
        """
        Divide the data into a training set and a test set.
        """
        X = self.data.drop(['TotalClaims', 'TotalPremium'], axis=1)
        y_claims = self.data['TotalClaims']
        y_premium = self.data['TotalPremium']
        
        X_train, X_test, y_claims_train, y_claims_test = train_test_split(X, y_claims, test_size=0.2, random_state=42)
        X_train_premium, X_test_premium, y_premium_train, y_premium_test = train_test_split(X, y_premium, test_size=0.3, random_state=42)
        
        return X_train, X_test, y_claims_train, y_claims_test, X_train_premium, X_test_premium, y_premium_train, y_premium_test
    

    def build_models(self, X_train, y_train):
        """
        Implement Linear Regression, Decision Trees, Random Forests, and XGBoost models.
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor()
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluate each model using appropriate metrics.
        """
        results = {}
        for name, model in models.items():
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results[name] = {'MSE': mse, 'R2': r2}
        
        return results

    def analyze_feature_importance(self, model, X_train, feature_names):
        """
        Analyze feature importance using SHAP for tree-based models like XGBoost.
        """
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
        
        # Plot summary
        shap.summary_plot(shap_values, X_train, feature_names=feature_names)

        # Show feature importance bar plot
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar")

    
    