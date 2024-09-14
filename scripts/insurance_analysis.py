import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Insurance_EDA:
    def __init__(self, df):
        """
        Initialize with the dataframe.
        """
        self.df = df
        
    
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
        missing_percentage = self.df.isnull().mean()
        cols_to_drop = missing_percentage[missing_percentage > null_threshold].index
        self.df = self.df.drop(columns=cols_to_drop)
        
        # 2. Drop low and high cardinality columns
        cardinality = self.df.select_dtypes(include="object").nunique()
        low_cardinality_cols = cardinality[cardinality < low_cardinality_threshold].index
        high_cardinality_cols = cardinality[cardinality > high_cardinality_threshold].index
        self.df = self.df.drop(columns=list(low_cardinality_cols) + list(high_cardinality_cols))
        
        # 3. Filter for outliers in numeric columns using the IQR method
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.10)
            Q3 = self.df[col].quantile(0.90)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            self.df = self.df[(self.df[col] >= Q1 - outlier_step) & (self.df[col] <= Q3 + outlier_step)]
        
        # 4. Fill missing numeric values with the mean
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        
        # 5. Fill missing categorical values with the mode
        categorical_cols = self.df.select_dtypes(include="object").columns
        self.df[categorical_cols] = self.df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col)
        
        return self.df

    def univariate_analysis(self):
        """
        Perform univariate analysis by plotting histograms for numerical columns 
        and bar charts for categorical columns.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns

        # Plotting histograms for numeric columns
        for col in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col].dropna(), bins=30, kde=True, color='skyblue')
            plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.show()

        # Plotting bar charts for categorical columns
        for col in categorical_columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.df[col], palette='Set2')
            plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.show()

    def bivariate_analysis(self):
        """
        Perform bivariate analysis by plotting scatter plots between TotalPremium and 
        TotalClaims with ZipCode, and generating a correlation matrix for numerical columns.
        """
        # Scatter plot for TotalPremium and TotalClaims by ZipCode
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='PostalCode', data =self.df, palette='viridis')
        plt.title('Total Premium vs. Total Claims by ZipCode', fontsize=14, fontweight='bold')
        plt.xlabel('TotalPremium', fontsize=12)
        plt.ylabel('TotalClaims', fontsize=12)
        plt.show()

        # Correlation matrix for numeric columns
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_columns.corr()

        # Plotting the correlation matrix using heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.show()

    def compare_trends(self, geography_column):
        """
        Compare the trends of insurance cover type, premium, and auto make over geography.
        :param geography_column: Column to compare the trends over (e.g., 'Province', 'PostalCode')
        """
        # Check if geography column exists in the dfset
        if geography_column not in self.df.columns:
            raise ValueError(f"Column '{geography_column}' not found in df.")
        
        # Group by the geography column and analyze trends
        geography_group = self.df.groupby(geography_column)

        # 1. Trends in Insurance Cover Type (CoverType)
        plt.figure(figsize=(10, 6))
        cover_type_counts = geography_group['CoverType'].value_counts().unstack().fillna(0)
        cover_type_counts.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
        plt.title(f'Insurance Cover Type Distribution by {geography_column}', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 2. Trends in Premiums (TotalPremium)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=geography_column, y='TotalPremium', data=self.df, palette='Set3')
        plt.title(f'Total Premiums by {geography_column}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 3. Trends in Auto Make (make)
        plt.figure(figsize=(10, 6))
        auto_make_counts = geography_group['make'].value_counts().unstack().fillna(0)
        top_auto_makes = auto_make_counts.sum().nlargest(10).index  # Get top 10 most frequent auto makes
        sns.countplot(x='make', hue=geography_column, data=self.df[self.df['make'].isin(top_auto_makes)], palette='Paired')
        plt.title(f'Top 10 Auto Makes by {geography_column}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def detect_outliers(self, numeric_columns=None):
        """
        This function plots box plots for numerical columns to detect outliers.
        
        Parameters:
        - numeric_columns: list of numerical columns to plot (optional).
        If None, it will plot all numerical columns in the dfFrame.
        """
        # If specific numeric columns are not provided, use all numeric columns in the dfset
        if numeric_columns is None:
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        # Plot box plots for each numeric column
        for col in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[col], color='skyblue')
            plt.title(f'Boxplot for {col}', fontsize=14, fontweight='bold')
            plt.show()

    
