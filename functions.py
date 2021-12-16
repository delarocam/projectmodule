"""Functions for analysis of agricultural yield
with diff in diff"""
from sklearn.model_selection import cross_validate
import sklearn.ensemble
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
# from sklearn.ensemble import RandomForestRegressor


def rain_frame():
    """cleans and modifies rain data for merge"""
    rain_df = pd.read_csv("aquastat.csv")

    rain_df = rain_df.loc[~(rain_df == 0).all(axis=1)]

    rain_df = rain_df[["Area", "Value"]]

    # making the data based on standard deviations from global mean 990 mm
    # to avoid collinearity with intercept

    rain_df["Value"] = (rain_df["Value"] - 990) / rain_df["Value"].std()

    rain_df = rain_df.rename(columns={"Value": "rain_stdev"})
    return rain_df


def temp_frame():
    """cleaning and returning yearly
    temperature data fro merge"""

    temp_df = pd.read_csv("tempstat.csv")

    temp_df = temp_df[["dt", "Country", "AverageTemperature"]]

    temp_df['dt'] = pd.to_datetime(temp_df['dt'])

    temp_df['dt'] = pd.DatetimeIndex(temp_df['dt']).year

    temp_df = temp_df.rename(columns={"Country": "Area", "dt": "Year"})

    temp_df = temp_df.dropna()

    # averaging out temperature by year by country

    temp_df = temp_df.groupby(["Area", "Year"], as_index=False)[
                               "AverageTemperature"].mean()

    # filtering by year for time of interest

    temp_df = temp_df[temp_df["Year"] >= 1990]

    return temp_df


def yield_frame():
    """cleaning and returning crop yield
    by country for merge"""

    yield_df = pd.read_csv("FAOSTAT_data_croptype.csv")

    yield_df = yield_df.rename(columns={"Value": "Yield_hg_ha"})

    yield_df = yield_df[["Area", "Year", "Item", "Yield_hg_ha"]]

    # consolidating crop types into average yield

    yield_df = yield_df.groupby(["Area", "Year"],
                                as_index=False)["Yield_hg_ha"].mean()
    return yield_df


# wrangling land_df for merge
def land_frame():
    """cleaning and returning crop area
    data for merge"""

    land_df = pd.read_csv("FAOSTAT_data_croparea.csv")

    land_df = land_df.rename(columns={"Value": "ha_cropland"})

    land_df = land_df[["Area", "Year", "ha_cropland"]]

    land_df["ha_cropland"] = land_df["ha_cropland"] * 1000

    return land_df


def fert_frame():
    """cleaning and returning fertilizer
    data for merge with other frames"""

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
                                         "kg_nitrogen",
                                         "Nutrient phosphate P2O5 (total)":
                                         "kg_phosphate",
                                         "Nutrient potash K2O (total)":
                                         "kg_potash"})

    # data expressed in 1000s of kilograms so multiplying by 1000
    # to modify by kilograms
    fertilizer_df[["kg_nitrogen", "kg_phosphate",
                   "kg_potash"]] = fertilizer_df[
                                       ["kg_nitrogen",
                                        "kg_phosphate",
                                        "kg_potash"]] * 1000
    return fertilizer_df


def manure_frame():
    """importing and cleaning manure data for merge"""

    manure_df = pd.read_csv("FAOSTAT_data_manure.csv")

    manure_df = manure_df.rename(columns={"Value": "kg_manure"})

    manure_df = manure_df[["Area", "Year", "kg_manure"]]

    return manure_df


def pest_frame():
    """importing and cleaning pesticides data
    for merge"""

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

    # data expressed in thousands of kg , so normalizing to kg

    pesticides_df[["kg_Fungicides_Bactericides", "kg_Herbicides",
                   "kg_Insecticides"]] = pesticides_df[
                                        ["kg_Fungicides_Bactericides",
                                         "kg_Herbicides",
                                         "kg_Insecticides"]] * 1000
    return pesticides_df.dropna()


def gni_frame():
    """importing and cleaning gni data for merge,
    setting up dummy variablesw for post 2008 and
    is_wealthy"""

    gni_df = pd.read_csv("gni_data.csv")

    gni_df = gni_df.rename(columns={"Country Name": "Area"})

    keys = [c for c in gni_df if c.startswith('1') or c.startswith('2')]

    gni_df = pd.melt(gni_df, id_vars='Area', value_vars=keys,
                     value_name='GNI')
    gni_df = gni_df.rename(columns={"variable": "Year"})

    gni_df["Year"] = pd.to_numeric(gni_df["Year"])

    gni_df = gni_df.loc[gni_df['Year'] >= 1990]

    gni_df = gni_df.loc[gni_df['Year'] <= 2013]

    gni_df.dropna()

    gni_df["is_wealthy"] = 0

    # world bank wealth level for "high income"
    # is set at around 12000$ per capita

    gni_df["is_wealthy"] = gni_df.is_wealthy.where(gni_df.GNI <= 12000, 1)

    gni_df["post_2008"] = 0

    gni_df["post_2008"] = gni_df.post_2008.where(gni_df.Year < 2008, 1)

    return gni_df.dropna()


def wrangle_frame():
    """merging all above data into one df
    also defining local function to merge all
    data together by Area and Year"""

    def merger_fun(dataframe1, dataframe2):
        final_df = dataframe1.merge(dataframe2,
                                    on=["Area", "Year"], how="left")
        return final_df

    # final_agro_df = merger_fun(weather_df, yield_df)

    yield_df = yield_frame()

    land_df = land_frame()

    fertilizer_df = fert_frame()

    manure_df = manure_frame()

    pesticides_df = pest_frame()

    temp_df = temp_frame()

    gni_df = gni_frame()

    rain_df = rain_frame()

    final_agro_df = yield_df

    final_agro_df = merger_fun(final_agro_df, land_df)

    final_agro_df = merger_fun(final_agro_df, fertilizer_df)

    final_agro_df = merger_fun(final_agro_df, manure_df)

    final_agro_df = merger_fun(final_agro_df, pesticides_df)

    final_agro_df = merger_fun(final_agro_df, temp_df)

    final_agro_df = merger_fun(final_agro_df, gni_df)

    final_agro_df = final_agro_df.merge(rain_df,
                                        on=["Area"], how="right")
    column_list = ["kg_Fungicides_Bactericides",
                   "kg_Herbicides", "kg_Insecticides",
                   "kg_nitrogen", "kg_phosphate",
                   "kg_potash", "kg_manure"]

    for i in column_list:
        final_agro_df[i] = final_agro_df[i] / final_agro_df["ha_cropland"]
        final_agro_df = final_agro_df.rename(columns={i: i + str("_ha")})

    final_agro_df = final_agro_df.drop_duplicates(subset=[
                                                  "AverageTemperature"])

    final_agro_df = final_agro_df.rename(columns={"Yield_hg_ha":
                                                  "Total_Yield"})

    final_agro_df["Total_Yield"] = final_agro_df[
                                   "Total_Yield"] * final_agro_df[
                                                    "ha_cropland"]

    final_agro_df["Log_Yield"] = np.log(final_agro_df["Total_Yield"])
    return final_agro_df.dropna()


def fung_fert_time_plot(dataframe):
    """returns informative graphs exploring the
    relationship between agricultural yield and inputs
    expressed over time"""
    cleaned_data = dataframe

# charting mean pesticide use over the years
    plt.figure(1)

    pest_columns = ["kg_Insecticides_ha", "kg_Herbicides_ha",
                    "kg_Fungicides_Bactericides_ha"]

    pest_plot_df = cleaned_data[["Year", "kg_Insecticides_ha",
                                "kg_Herbicides_ha",
                                 "kg_Fungicides_Bactericides_ha"]]
    pest_plot_df = pest_plot_df.groupby("Year",
                                        as_index=False)[
                                        pest_columns].mean()

    for i in range(3):
        plt.plot(pest_plot_df["Year"], pest_plot_df[pest_columns[i]],
                 label=pest_columns[i])
    plt.ylabel("average kg/ha")
    plt.xlabel("year")
    plt.title("mean pesticide use 1990-2013")
    plt.legend(loc='upper left')

    plt.savefig("pesticide_use")

    plt.clf()

# charting mean fertilizer use over the years
    plt.figure(2)

    fert_columns = ["kg_nitrogen_ha", "kg_phosphate_ha",
                    "kg_potash_ha", "kg_manure_ha"]

    fert_plot_df = cleaned_data[["Year", "kg_nitrogen_ha",
                                 "kg_phosphate_ha",
                                "kg_potash_ha", "kg_manure_ha"]]

    fert_plot_df = fert_plot_df.groupby("Year",
                                        as_index=False)[
                                        fert_columns].mean()

    for i in range(4):
        plt.plot(fert_plot_df["Year"], fert_plot_df[fert_columns[i]],
                 label=fert_columns[i])
    plt.ylabel("average kg/ha")
    plt.xlabel("year")
    plt.title("mean fertilizer use 1990-2013")
    plt.legend(loc='upper left')

    plt.savefig("fertilizer_use")

    plt.clf()


def yield_time_plot(data_frame):
    """plots for yeild over time"""
# charting percent change of crop yield over years,
    plt.figure(3)
    cleaned_data = data_frame
    crop_yield_df = cleaned_data
    crop_yield_np = crop_yield_df.groupby("Area",
                                          as_index=False)[
                                          "Total_Yield"].pct_change()
    crop_yield_np = crop_yield_np.dropna()

    crop_yield_np = pd.merge(crop_yield_df, crop_yield_np, left_index=True,
                             right_index=True)

    crop_yield_np = crop_yield_np.set_index(["Area",
                                            "Year"])["Total_Yield_y"].unstack()

    crop_yield_np = crop_yield_np.dropna()

    crop_yield_np = crop_yield_np.reset_index()
    index_heat = crop_yield_np["Area"]
    cols_heat = list()
    for i in list(range(1991, 2013)):
        cols_heat.append(i)

    crop_yield_np = crop_yield_np[cols_heat].to_numpy()

    crop_yield_np = pd.DataFrame(crop_yield_np, index=index_heat,
                                 columns=cols_heat)

    sns.heatmap(data=crop_yield_np)

    plt.savefig("crop_heat")

    plt.clf()

    # plotting global cropyield by year

    plt.figure(4)

    crop_yield_ag = crop_yield_df.groupby("Year",
                                          as_index=False)["Total_Yield"].mean()

    plt.plot(crop_yield_ag["Year"], crop_yield_ag["Total_Yield"],
             label="Total_Yield_kg")

    plt.ylabel("total_avg_Yield kg_e^12")
    plt.xlabel("year")
    plt.title("Total_avg_Yield 1990-2013")
    plt.legend(loc='upper left')

    plt.savefig("total_yield")

    plt.clf()

    plt.figure(5)

    crop_yield_wealth = crop_yield_df.groupby(["Year", "is_wealthy"],
                                              as_index=False)[
                                              "Total_Yield"].mean()

    crop_yield_poor = crop_yield_wealth[crop_yield_wealth["is_wealthy"] < 1]
    crop_yield_rich = crop_yield_wealth[crop_yield_wealth["is_wealthy"] >= 1]
    plt.plot(crop_yield_poor["Year"], crop_yield_poor["Total_Yield"],
             label="country_gni < 12000")

    plt.plot(crop_yield_rich["Year"], crop_yield_rich["Total_Yield"],
             label="country_gni >= 12000")

    plt.ylabel("total_avg_Yield kg_e^12")
    plt.xlabel("year")
    plt.title("Total avg Yield by wealth level 1990-2013")
    plt.legend(loc='upper left')

    plt.savefig("total_yield_rich")

    plt.clf()


def scatter_plots(dataframe):
    """plotting basic scatterplots between vars
    to see if there are any interesting relationships"""
# plotting percapita gdp against log_yeild, to see
# if richer countries generally have larger yields
    cleaned_data = dataframe
    plt.figure(6)

    plt.scatter(cleaned_data["GNI"], cleaned_data["Log_Yield"])

    plt.title("GNI vs Log Crop Yield")

    plt.xlabel("GNI")

    plt.ylabel("Log Yield")

    plt.savefig("GNI_yield")
    plt.clf()
# plotting insecticide use vs log_yield
    plt.figure(7)

    plt.scatter(cleaned_data["kg_Insecticides_ha"], cleaned_data["Log_Yield"])

    plt.title("Insecticide use vs Crop Yield")

    plt.xlabel("Insecticides kg/ha")

    plt.ylabel("Log_Yield")

    plt.savefig("insecticide_yield")
    plt.clf()
# plotting herbicide use vs log_yield
    plt.figure(8)

    plt.scatter(cleaned_data["kg_Herbicides_ha"], cleaned_data["Log_Yield"])

    plt.title("Herbecide use vs Log Crop Yield")

    plt.xlabel("Herbicide kg/ha")

    plt.ylabel("Log Yield")

    plt.savefig("herbicide_yield")
    plt.clf()
# plotting fungicide use vs log_yield
    plt.figure(9)

    plt.scatter(cleaned_data["kg_Fungicides_Bactericides_ha"],
                cleaned_data["Log_Yield"])

    plt.title("Fungicide/Bactericide use vs Log Crop Yield")

    plt.xlabel("Fungicide/Bactericide kg/ha")

    plt.ylabel("Log Yield")

    plt.savefig("Fung_bac_yield")
    plt.clf()
# plotting manure use vs gdp
    plt.figure(10)

    plt.scatter(cleaned_data["AverageTemperature"], cleaned_data[
                            "rain_stdev"])

    plt.title("temp vs rain")

    plt.xlabel("temp")

    plt.ylabel("rain")

    plt.savefig("rain_temp_gdp")

    plt.clf()


# printing out summary stats
def summary_stats(numeric_vector):
    """"returns summary statistics for
    each relevant numeric column in the dataframe,
    in order to better understand the distribution
    of data, called for each column and returns a list,
    of mean, st. dev, variance, and median"""
    cleaned_data = wrangle_frame()
    summary_list = []
    summary_list.append(cleaned_data[numeric_vector].mean(numeric_only=True))
    summary_list.append(cleaned_data[numeric_vector].std(numeric_only=True))
    summary_list.append(cleaned_data[numeric_vector].var(numeric_only=True))
    summary_list.append(cleaned_data[numeric_vector].median(numeric_only=True))

    return summary_list
# define column vector to summarize


# ols model function
def ols_model(dataframe):
    """runs an ols regression predicting yield,
       based on relevent variables such as tempurature,
       pesticide use, rainfall , and other relevent variables.
       uses training and test sets to cross validate.
       returns model and prediction accuracy"""
# note rainfall, tempurature are collinear with eachother, gdp_per cap is
# collinear with temp,
# insecticide use is collinear with rainfall, due to tropics, what to do ?
    cleaned_data = dataframe

    diff_means_df = cleaned_data.groupby(["is_wealthy", "Year"],
                                         as_index=False)["Total_Yield"].mean()

    diff_means_df["post_2008"] = 0

    diff_means_df["post_2008"] = diff_means_df.post_2008.where(
                                 diff_means_df.Year < 2008, 1)

    diff_means_df = diff_means_df.set_index(["Year",
                                             "is_wealthy",
                                             "post_2008"]).unstack(
                                            "is_wealthy")

    diff_means_df = diff_means_df.groupby("post_2008").mean()

    print(diff_means_df)

    smf_model = smf.ols(formula="Log_Yield~is_wealthy + post_2008 +" +
                        str("post_2008*is_wealthy + rain_stdev +") +
                        str("AverageTemperature + kg_Insecticides_ha") +
                        str(" + kg_Fungicides_Bactericides_ha") +
                        str(" + kg_Herbicides_ha + kg_potash_ha") +
                        str(" + kg_nitrogen_ha + kg_phosphate_ha") +
                        str("+ kg_manure_ha"), data=cleaned_data)

    return smf_model


# might test out a boosting and bagging model as well


def random_forest_model(dataframe):
    """runs a more flexible model, random forest
       in order to predict crop yield
       and the relevent variables.
       uses training and test sets to k-fold cross validate.
       returns random forest model and prediction accuracy"""
    cleaned_data = dataframe
    feature_list = ["kg_nitrogen_ha", "kg_phosphate_ha",
                    "kg_potash_ha", "kg_manure_ha",
                    "kg_Fungicides_Bactericides_ha",
                    "kg_Herbicides_ha", "kg_Insecticides_ha",
                    "AverageTemperature",
                    "rain_stdev", "is_wealthy",
                    "post_2008"]

    x_var = cleaned_data[feature_list]

    x_columns = x_var.columns

    # truncates feature names to more easily see on
    # graph

    for i in range(len(feature_list)):
        feature_list[i] = feature_list[i][0:5]

    # normalizing data

    x_var = preprocessing.normalize(x_var)

    y_var = cleaned_data[["Log_Yield"]].values.ravel()

    forest = sklearn.ensemble.RandomForestRegressor(random_state=5,
                                                    n_estimators=500,
                                                    max_depth=5)

    cv_mod = cross_validate(forest, x_var, y_var, cv=5,
                            scoring="neg_mean_absolute_percentage_error")

    mean_per = cv_mod["test_score"].mean()

    print("test score array")

    print(cv_mod["test_score"])

    print("Mean percentage error")

    print(mean_per)

    forest.fit(x_var, y_var)

    feat_importances = pd.Series(forest.feature_importances_,
                                 index=x_columns)
    plt.figure(11)

    feat_importances.nlargest(11).plot(kind='barh')

    plt.title("relative importance")

    plt.savefig("importance_plot")

    return mean_per


def main():
    """function desinged to call all module functions"""
    wrangled = wrangle_frame()

    fung_fert_time_plot(wrangled)

    yield_time_plot(wrangled)

    scatter_plots(wrangled)

    columns_to_summarize = ["Total_Yield", "ha_cropland",
                            "kg_nitrogen_ha", "kg_phosphate_ha",
                            "kg_potash_ha", "kg_manure_ha",
                            "kg_Fungicides_Bactericides_ha",
                            "kg_Herbicides_ha", "kg_Insecticides_ha",
                            "GNI", "AverageTemperature",
                            "rain_stdev", "Log_Yield"]

    summary = summary_stats(columns_to_summarize)
    print()
    print("Column Means")
    print()
    print(summary[0])
    print()
    print("Column st.devs")
    print()
    print(summary[1])
    print()
    print("Column variances")
    print()
    print(summary[2])
    print()
    print("Column medians")
    print()
    print(summary[3])
    print()

    model_1 = ols_model(wrangled)

    model_fit = model_1.fit()

    print(model_fit.summary())

    random_forest_model(wrangled)


main()
