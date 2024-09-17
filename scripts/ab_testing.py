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
    

