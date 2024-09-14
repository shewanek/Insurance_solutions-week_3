# scripts/eda_functions.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Insurance_EDA:
    def __init__(self, dataframe):
        """
        Initialize with the dataframe to be used for EDA.
        """
        self.df = dataframe
    
    