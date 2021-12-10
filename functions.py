"""Functions for analysis of agricultural yield"""
# from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import statsmodels as sms
# from sklearn.ensemble import RandomForestRegressor

# wrangling weather_df for merge


def wrangle_frame():

    weather_df = pd.read_csv("yield_df.csv")

    weather_df = weather_df[["Area", "Year",
                            "average_rain_fall_mm_per_year",
                             "avg_temp"]]

    weather_df = weather_df.sort_values(['Area', 'Year'], ascending=True)

    weather_df = weather_df.groupby(['Area', 'Year']).first().reset_index()

    print(weather_df)

# print(weather_df.head())

# wrangling yield_df for merge

    yield_df = pd.read_csv("FAOSTAT_data_croptype.csv")

    yield_df = yield_df.rename(columns={"Value": "Yield_hg\ha"})

    yield_df = yield_df[["Area", "Year", "Item", "Yield_hg\ha"]]

    yield_df = yield_df.groupby(["Area", "Year"],
                                as_index=False)["Yield_hg\ha"].sum()

# print(yield_df.head())

# wrangling land_df for merge

    land_df = pd.read_csv("FAOSTAT_data_croparea.csv")

    land_df = land_df.rename(columns={"Value": "ha_cropland"})

    land_df = land_df[["Area", "Year", "ha_cropland"]]

    land_df["ha_cropland"] = land_df["ha_cropland"] * 1000

# print(land_df.head())

# wrangling fertilizer_df for merge

    fertilizer_df = pd.read_csv("FAOSTAT_data_fertilizer.csv")

    fertilizer_df = fertilizer_df.rename(columns={"Value": "tonnes_used"})

    fertilizer_df = fertilizer_df[["Area", "Year", "Item", "tonnes_used"]]

    fertilizer_df = fertilizer_df.set_index(['Area', 'Year',
                                             'Item'])['tonnes_used'].unstack()

# when unstacked set 0 to NA but was 0 in original dataframe

    fertilizer_df = fertilizer_df[["Nutrient nitrogen N (total)",
                                   "Nutrient phosphate P2O5 (total)",
                                   "Nutrient potash K2O (total)"]].fillna(0)

    fertilizer_df = fertilizer_df.reset_index()

    fertilizer_df = fertilizer_df.rename(columns={
                                         "Nutrient nitrogen N (total)":
                                         "kg_nitrogen_fertilizer",
                                         "Nutrient phosphate P2O5 (total)":
                                         "kg_phosphate_fertilizer",
                                         "Nutrient potash K2O (total)":
                                         "kg_potash_fertilizer"})

    fertilizer_df[["kg_nitrogen_fertilizer", "kg_phosphate_fertilizer",
                   "kg_potash_fertilizer"]] = fertilizer_df[
                                       ["kg_nitrogen_fertilizer",
                                        "kg_phosphate_fertilizer",
                                        "kg_potash_fertilizer"]] * 1000
# print(fertilizer_df.head())


# wrangling manure_df for merge
    manure_df = pd.read_csv("FAOSTAT_data_manure.csv")

    manure_df = manure_df.rename(columns={"Value": "kg_manure"})

    manure_df = manure_df[["Area", "Year", "kg_manure"]]

# wrangling pesticides_df for merge

    pesticides_df = pd.read_csv("FAOSTAT_data_pesticides.csv")

    pesticides_df = pesticides_df.rename(columns={"Value": "tonnes_used"})

    pesticides_df = pesticides_df[["Area", "Year", "Item", "tonnes_used"]]

    pesticides_df = pesticides_df.set_index(['Area', 'Year',
                                             'Item'])['tonnes_used'].unstack()

    pesticides_df = pesticides_df.reset_index()

    pesticides_df = pesticides_df.rename(columns={
                                        "Fungicides and Bactericides":
                                        "kg_Fungicides_Bactericides",
                                        "Herbicides":
                                        "kg_Herbicides",
                                        "Insecticides":
                                        "kg_Insecticides"})

    pesticides_df[["kg_Fungicides_Bactericides", "kg_Herbicides",
                   "kg_Insecticides"]] = pesticides_df[
                                        ["kg_Fungicides_Bactericides",
                                         "kg_Herbicides",
                                         "kg_Insecticides"]] * 1000
# print(pesticides_df.head())
# wrangling gdp_df for merge

    gdp_df = pd.read_csv("FAOSTAT_data_gdp.csv")

    gdp_df = gdp_df[["Area", "Year", "Value"]]

    gdp_df = gdp_df.rename(columns={"Value": "percap_gdp($)"})

# merging dataframes

    def merger_fun(dataframe1, dataframe2):
        final_df = dataframe1.merge(dataframe2,
                                    on=["Area", "Year"], how="left")
        return final_df

    final_agro_df = merger_fun(weather_df, yield_df)

    final_agro_df = merger_fun(final_agro_df, land_df)

    final_agro_df = merger_fun(final_agro_df, fertilizer_df)

    final_agro_df = merger_fun(final_agro_df, manure_df)

    final_agro_df = merger_fun(final_agro_df, pesticides_df)

    final_agro_df = merger_fun(final_agro_df, gdp_df)

    column_list = ["kg_Fungicides_Bactericides",
                   "kg_Herbicides", "kg_Insecticides",
                   "kg_nitrogen_fertilizer", "kg_phosphate_fertilizer",
                   "kg_potash_fertilizer", "kg_manure"]

    for i in column_list:
        final_agro_df[i] = final_agro_df[i] / final_agro_df["ha_cropland"]
        final_agro_df = final_agro_df.rename(columns={i: i + str("\ha")})
    return final_agro_df


cleaned_data = wrangle_frame()

plt.figure(1)

pesticide_line_plot = 

def graph_agri():


    """returns informative graphs exploring the
    relationship between different variables that
    could effect agricultural yeild, including tempurature,
    rainfall, pesticide use, and crop type"""


def summary_stats(column_name):
    """"returns summary statistics for
    each relevant column in the dataframe,
    in order to better understand the distribution
    of data, called for each column and returns a list,
    of mean, st. dev, variance, and median"""
    return column_name


def ols_model(dataframe):
    """runs an ols regression predicting yield,
       based on relevent variables such as tempurature,
       pesticide use, rainfall , and other relevent variables.
       uses training and test sets to k-fold cross validate.
       returns model and prediction accuracy"""
    return dataframe


# might test out a boosting and bagging model as well

def random_forest_model(dataframe):
    """runs a more flexible model, random forest
       in order to predict crop yield
       and the relevent variables.
       uses training and test sets to k-fold cross validate.
       returns random forest model and prediction accuracy"""
    return dataframe
