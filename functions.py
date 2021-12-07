"""Functions for analysis of agricultural yield"""
from pandas.core.frame import DataFrame
# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# import statsmodels as sms
# from sklearn.ensemble import RandomForestRegressor

test_df = pd.read_csv("yield_df.csv")


def wrangle_frame(dataframe: DataFrame) -> DataFrame:
    """function intended to merge the required
     csv files "yield", "rainfall", "pesticides" and "temp",
      and filter out unneeded columns
     and rows with missing data. I plan to make
     dummy variables for each crop type, and lagged
     variables for rain if possible to see if multiyear droughts
     or high rain has an effect"""
    pass


def graph_agri(dataframe: DataFrame):
    """returns informative graphs exploring the
    relationship between different variables that
    could effect agricultural yeild, including tempurature,
    rainfall, pesticide use, and crop type"""
    pass


def summary_stats(column_name: str) -> list:
    """"returns summary statistics for
    each relevant column in the dataframe,
    in order to better understand the distribution
    of data, called for each column and returns a list,
    of mean, st. dev, variance, and median"""
    pass


def ols_model():
    """runs an ols regression predicting yield,
       based on relevent variables such as tempurature,
       pesticide use, rainfall , and other relevent variables.
       uses training and test sets to k-fold cross validate.
       returns model and prediction accuracy"""
    pass


# might test out a boosting and bagging model as well

def random_forest_model():
    """runs a more flexible model, random forest
       in order to predict crop yield
       and the relevent variables.
       uses training and test sets to k-fold cross validate.
       returns random forest model and prediction accuracy"""
    pass
