---

# AlphaCare Insurance Solutions: Car Insurance Claim Analysis

## Project Overview

AlphaCare Insurance Solutions (ACIS) aims to enhance its car insurance planning and marketing strategies through predictive analytics and risk assessment. The primary objective of this project is to analyze historical insurance claims data, identify low-risk clients, and suggest ways to optimize marketing efforts. This analysis will help ACIS better tailor its insurance products and identify opportunities for premium reductions to attract new customers.

As a **Marketing Analytics Engineer** for ACIS, my role is to perform data analysis, statistical testing, and machine learning modeling to uncover actionable insights. These insights will help differentiate risk levels across various client groups, regions, and vehicle types.

## Business Objectives

1. **Optimize Marketing Strategy**  
   Analyze the historical insurance claims data to help develop marketing strategies that attract low-risk customers.
   
2. **Identify Low-Risk Customers**  
   Discover customer segments where premiums could be reduced without significant risk, providing a competitive advantage to ACIS.

3. **Statistical Testing**  
   Use A/B hypothesis testing to identify significant differences in risk profiles across regions, zip codes, and demographics (gender, age, etc.).

4. **Predictive Modeling**  
   Build machine learning models to predict optimal premiums based on car features, client demographics, location, and other factors.

## Data Description

The historical data spans from **February 2014 to August 2015**, covering detailed information on policies, clients, vehicles, and claims. Below is a high-level overview of the data structure:

### **Policy Information**
- `UnderwrittenCoverID`, `PolicyID`, `TransactionMonth`

### **Client Information**
- `IsVATRegistered`, `Citizenship`, `LegalType`, `MaritalStatus`, `Gender`, etc.

### **Location Information**
- `Country`, `Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`

### **Car Information**
- `VehicleType`, `Make`, `Model`, `RegistrationYear`, `NumberOfDoors`, `CubicCapacity`

### **Insurance Plan Information**
- `SumInsured`, `CalculatedPremiumPerTerm`, `CoverCategory`, `CoverType`, `Product`

### **Payment and Claim Information**
- `TotalPremium`, `TotalClaims`

## Key Analyses & Methodologies

### 1. **Exploratory Data Analysis (EDA)**
   - Data visualization and statistical summaries to understand distribution, anomalies, and relationships between features.

### 2. **A/B Hypothesis Testing**
   - Test the following hypotheses:
     - No risk differences across provinces.
     - No significant margin differences across zip codes.
     - No significant risk differences between genders.

### 3. **Predictive Modeling**
   - Develop a linear regression model to predict total claims per zip code.
   - Use machine learning algorithms to predict optimal premium values based on:
     - Features of the insured car.
     - Owner demographics and location.

### 4. **Feature Importance**
   - Analyze which features most influence the premium prediction model, such as vehicle make, model, location, and owner's profile.

## Tools & Technologies

The project leverages several Python packages and tools for analysis and model building:
- **Python**: Primary programming language for the analysis.
- **Pandas & Numpy**: For data manipulation and exploration.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning modeling.
- **Statistical Testing**: Chi-squared tests, logistic regression, and t-tests for hypothesis testing.
- **DVC**: For data versioning and experiment tracking.
- **GitHub & Git**: For version control, CI/CD, and collaboration.

## Project Workflow

1. **Data Cleaning and Preprocessing**
   - Handle missing data, ensure data consistency, and transform categorical variables.

2. **Exploratory Data Analysis (EDA)**
   - Gain insights into the data, understand distributions, and identify potential relationships.

3. **Hypothesis Testing**
   - Perform A/B tests to evaluate risk differences across key demographics and regions.

4. **Machine Learning Model Development**
   - Train and validate models to predict claims and premiums.
   - Evaluate model performance using metrics such as MAE, RMSE, and R².

5. **Final Report and Recommendations**
   - Present findings, insights, and actionable recommendations to optimize ACIS's car insurance offerings.

## Deliverables

- **Data Analysis Report**: Summary of key findings from the data analysis, including visualizations and statistical insights.
- **Machine Learning Models**: Predictive models for risk and premium determination.
- **Recommendations**: Suggestions for premium adjustments and marketing strategies based on low-risk targets identified from the data.

---

## Installation & Setup

### 1. **Clone the repository**
   ```bash
   git clone https://github.com/shewanek/Insurance_solutions-week_3.git
   ```

### 2. **Install the required Python packages**
   ```bash
   pip install -r requirements.txt
   ```

### 3. **Run the analysis**
   Open the Jupyter notebooks:
   ```bash
   jupyter notebook eda.ipynb
   ```

---

## Contact

For any questions or issues, please contact [zshewanek@gmail.com](mailto:zshewanek@gmail.com).

Connect with me on [LinkedIn](https://www.linkedin.com/in/shewanek-zewdu/).

---

