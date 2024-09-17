import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class ABHypothesisTesting:
    def __init__(self, df):
        """
        Initialize with the dataframe.
        """
        self.data = df
    
    def select_metrics(self):
        """
        Here the key performance indicator (KPI) that will measure the impact of the features being tested.
        """
        return ['TotalClaims', 'TotalPremium']

    def segment_data(self, feature, group_a, group_b):
        """
        Segment the data into two groups: Control (A) and Test (B) based on the feature.
        :param feature: Feature to segment (e.g., 'Province', 'Gender').
        :param group_a: Value for Group A (Control).
        :param group_b: Value for Group B (Test).
        :return: Two dataframes representing Group A and Group B.
        """
        group_a_data = self.data[self.data[feature] == group_a]
        group_b_data = self.data[self.data[feature] == group_b]
        return group_a_data, group_b_data

    def perform_t_test(self, group_a_data, group_b_data, metric):
        """
        Perform T-test between Group A and Group B on a numeric metric.
        :param group_a_data: Data for Group A.
        :param group_b_data: Data for Group B.
        :param metric: Column to test (e.g., 'TotalClaims', 'ProfitMargin').
        """
        t_stat, p_value = stats.ttest_ind(group_a_data[metric], group_b_data[metric], nan_policy='omit')
        return t_stat, p_value

    def perform_chi_squared_test(self, feature):
        """
        Perform Chi-squared test on categorical features.
        :param feature: Categorical feature (e.g., 'Province', 'Gender').
        """
        contingency_table = pd.crosstab(self.data['TotalClaims'], self.data[feature])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2, p_value
    
    def analyze_results(self, p_value, alpha=0.05):
        """
        Analyze the p-value to accept/reject the null hypothesis.
        :param p_value: P-value from the statistical test.
        :param alpha: Significance level (default=0.05).
        """
        if p_value < alpha:
            return "Reject the null hypothesis (statistically significant)."
        else:
            return "Fail to reject the null hypothesis (not statistically significant)."

    def visualize_distribution(self, feature, metric):
        """
        Visualize the distribution of a metric (e.g., 'TotalClaims', 'TotalPremium') based on a categorical feature.
        :param feature: Categorical feature to compare (e.g., 'Province', 'Gender').
        :param metric: Numeric metric to visualize (e.g., 'TotalClaims', 'ProfitMargin').
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature, y=metric, data=self.data)
        plt.title(f'{metric} Distribution by {feature}')
        plt.xlabel(feature)
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_bar(self, feature, metric):
        """
        Visualize a bar chart to show the mean of a numeric metric grouped by a categorical feature.
        :param feature: Categorical feature (e.g., 'Province', 'Gender').
        :param metric: Numeric metric to visualize (e.g., 'TotalClaims', 'ProfitMargin').
        """
        plt.figure(figsize=(10, 6))
        group_means = self.data.groupby(feature)[metric].mean().reset_index()
        sns.barplot(x=feature, y=metric, data=group_means)
        plt.title(f'Mean {metric} by {feature}')
        plt.xlabel(feature)
        plt.ylabel(f'Mean {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_correlation(self, features):
        """
        Visualize the correlation matrix for a list of features.
        :param features: List of numeric features to include in the correlation matrix.
        """
        corr_matrix = self.data[features].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()