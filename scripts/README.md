---

# Insurance Data Analysis and Modeling

## Overview

This project focuses on analyzing and modeling insurance data to optimize marketing strategies and discover low-risk targets for premium adjustments. The project includes data wrangling, exploratory data analysis (EDA), hypothesis testing, and predictive modeling. Key steps involve cleaning and preprocessing the data, performing univariate and bivariate analyses, and building and evaluating various machine learning models.

## Components

### 1. `Insurance_EDA`
This class is responsible for performing Exploratory Data Analysis (EDA) on the insurance dataset. It includes methods for:
- **Data Wrangling**: Cleans and preprocesses the data by handling missing values, filtering outliers, and dropping irrelevant columns.
- **Univariate Analysis**: Plots histograms for numerical columns and bar charts for categorical columns.
- **Bivariate Analysis**: Creates scatter plots and correlation matrices to explore relationships between features.
- **Compare Trends**: Analyzes trends across different geographical regions.
- **Detect Outliers**: Visualizes outliers in numerical columns using box plots.
- **Visualize Key Insights**: Generates creative plots to capture key insights from the EDA.

### 2. `ABHypothesisTesting`
This class handles hypothesis testing to evaluate the impact of different features on key performance indicators (KPIs). It includes methods for:
- **Selecting Metrics**: Defines key performance indicators like Total Claims and Total Premium.
- **Segmenting Data**: Segments the data into control and test groups.
- **Performing T-tests**: Tests the difference between means of two groups.
- **Chi-squared Test**: Tests the independence of categorical features.
- **Analyzing Results**: Interprets p-values to make decisions about hypotheses.
- **Visualizing Distributions**: Plots distributions of metrics based on categorical features.
- **Visualizing Bar Charts**: Shows mean values of metrics grouped by categorical features.
- **Visualizing Correlation**: Displays a correlation matrix for numeric features.

### 3. `InsuranceModeling`
This class builds and evaluates predictive models for insurance claims and premiums. It includes methods for:
- **Data Wrangling**: Prepares the data by handling missing values, filtering outliers, and converting categorical variables.
- **Feature Engineering**: Creates new features to enhance model performance.
- **Encoding Categorical Data**: Converts categorical variables to numeric format using one-hot encoding.
- **Train-Test Split**: Splits the data into training and testing sets.
- **Building Models**: Trains multiple regression models including Linear Regression, Decision Trees, Random Forest, and XGBoost.
- **Evaluating Models**: Assesses model performance using metrics such as Mean Squared Error (MSE) and R-squared.
- **Analyzing Feature Importance**: Uses SHAP values to interpret the importance of features in tree-based models.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- XGBoost
- SHAP
- SciPy

Install the required packages using:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost shap scipy
```

## Usage

1. **Initialize and Load Data**: 
   Load your dataset into a Pandas DataFrame and initialize the classes with this data.

2. **Perform EDA**: 
   Use the `Insurance_EDA` class to clean and analyze the data.

3. **Hypothesis Testing**: 
   Use the `ABHypothesisTesting` class to test hypotheses and visualize results.

4. **Model Building**: 
   Use the `InsuranceModeling` class to prepare data, build models, evaluate them, and analyze feature importance.



## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues.
