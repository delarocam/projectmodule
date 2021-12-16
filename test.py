"""tests for functions in module"""
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.algorithms import quantile
from pandas.core.indexes import numeric
import functions as fn
import math
import unittest
from numpy.testing import assert_almost_equal


def test_rain_df():
    """test rain_df correct output"""
    rain_df = pd.read_csv("aquastat.csv")

    assert len(fn.rain_frame().columns) == 2

    assert len(fn.rain_frame()["rain_stdev"]) <= len(rain_df["Value"])

    assert len(fn.rain_frame().loc[~(fn.rain_frame() == 0).all(axis=1)][
           "rain_stdev"]) <= len(rain_df["Value"])

    assert len(fn.rain_frame().dropna()[
           "rain_stdev"]) == len(fn.rain_frame()["rain_stdev"])
    print("test_rain passed")


def test_temp():
    """test temp_df correct output"""
    temp_df = pd.read_csv("tempstat.csv")

    assert len(fn.temp_frame().columns) == 3

    assert len(fn.temp_frame()["AverageTemperature"]) <= len(temp_df[
                                                         "AverageTemperature"])

    assert len(fn.temp_frame().dropna()[
           "AverageTemperature"]) == len(fn.temp_frame()["AverageTemperature"])
    print("test_temp passed")


def test_yield():
    """test yield_df correct output"""
    yield_df = pd.read_csv("FAOSTAT_data_croptype.csv")

    assert len(fn.yield_frame().columns) == 3

    assert len(fn.yield_frame()["Yield_hg_ha"]) <= len(yield_df["Value"])

    assert len(fn.yield_frame().loc[~(fn.yield_frame() == 0).all(
                                 axis=1)]["Yield_hg_ha"]) == len(
                                fn.yield_frame()["Yield_hg_ha"])

    assert len(fn.yield_frame().dropna()[
           "Yield_hg_ha"]) == len(fn.yield_frame()["Yield_hg_ha"])

    print("test yield passed")


def test_land():
    """test land_df correct output"""
    land_df = pd.read_csv("FAOSTAT_data_croparea.csv")

    assert len(fn.land_frame().columns) == 3

    assert len(fn.land_frame()["ha_cropland"]) <= len(land_df["Value"])

    assert len(fn.land_frame().loc[~(fn.land_frame() == 0).all(
                                   axis=1)]["ha_cropland"]) == len(
                                   fn.land_frame()["ha_cropland"])

    assert len(fn.land_frame().dropna()[
           "ha_cropland"]) == len(fn.land_frame()["ha_cropland"])

    print("test land passed")


def test_fert():
    """test_fert_df correct ouput"""

    fert_df = pd.read_csv("FAOSTAT_data_fertilizer.csv")

    assert len(fn.fert_frame().columns) == 5

    assert len(fn.fert_frame()["kg_nitrogen"]) <= len(fert_df["Value"])

    assert len(fn.fert_frame().dropna()[
           "kg_nitrogen"]) == len(fn.fert_frame()["kg_nitrogen"])

    print("test land passed")


def test_manure():
    """test manure_df correct ouput"""
    manure_df = pd.read_csv("FAOSTAT_data_manure.csv")

    assert len(fn.manure_frame().columns) == 3

    assert len(fn.manure_frame()["kg_manure"]) <= len(manure_df["Value"])

    assert len(fn.manure_frame().dropna()[
           "kg_manure"]) == len(fn.manure_frame()["kg_manure"])

    print("test manure passed")


def test_pest():
    """test pest_df correct ouput"""
    pest_df = pd.read_csv("FAOSTAT_data_pesticides.csv")

    assert len(fn.pest_frame().columns) == 5

    assert len(fn.pest_frame()["kg_Herbicides"]) <= len(pest_df["Value"])

    assert len(fn.pest_frame().dropna()[
           "kg_Herbicides"]) == len(fn.pest_frame()["kg_Herbicides"])

    print("test pest passed")


def test_gni():
    gni_df = pd.read_csv("gni_data.csv")

    assert len(fn.gni_frame().columns) == 5

    assert len(fn.gni_frame()["GNI"]) >= len(gni_df["1990"])

    assert len(fn.gni_frame().dropna()[
           "GNI"]) == len(fn.gni_frame()["GNI"])

    assert max(fn.gni_frame()["post_2008"]) == 1

    print("test gni passed")


def test_data_wrangle():
    """will test the data_wrangle function
    to verify 16 columns """

    df_yield = fn.yield_frame()

    assert len(fn.wrangle_frame().columns) == 17

    assert max(fn.wrangle_frame()["Year"]) == 2013

    assert min(fn.wrangle_frame()["Year"]) == 1990

    assert len(fn.wrangle_frame()["Total_Yield"]) <= len(
               df_yield["Yield_hg_ha"])
    assert len(fn.wrangle_frame().dropna()["Total_Yield"]) == len(
               fn.wrangle_frame()["Total_Yield"])
    print("data wrangle tests passed")


def test_time_plot():
    """will test the graph_agri function
    in order to check that graphs are returned without
    error. Preliminary plans to make 6 plots"""

    def subtest_1():
        plt.figure(1)
        num_figures_before = plt.gcf().number
        fn.time_plot(fn.wrangle_frame())
        num_figures_after = plt.gcf().number
        assert num_figures_before < num_figures_after

    def subtest_2():
        fn.time_plot(fn.wrangle_frame())
        assert plt.gcf().number == 5

    subtest_1()

    subtest_2()

    print("time plot tests passed")


def test_scatter_plots():
    """will test the graph_agri function
    in order to check that graphs are returned without
    error. Preliminary plans to make 6 plots"""
    def subtest_1():
        fn.scatter_plots(fn.wrangle_frame())
        assert plt.gcf().number == 10

    def subtest_2():
        plt.figure(5)
        num_figures_before = plt.gcf().number
        fn.scatter_plots(fn.wrangle_frame())
        num_figures_after = plt.gcf().number
        assert num_figures_before < num_figures_after

    subtest_1()

    subtest_2()

    print("graphing tests passed")


def test_summary_stats():
    """Will test the summary_stats function,
    to make sure the summary stats are internally
    consitent and no data is being dropped or incorrectly
    summarized. """

    columns_to_summarize = ["Total_Yield", "ha_cropland",
                            "kg_nitrogen_ha", "kg_phosphate_ha",
                            "kg_potash_ha", "kg_manure_ha",
                            "kg_Fungicides_Bactericides_ha",
                            "kg_Herbicides_ha", "kg_Insecticides_ha",
                            "GNI", "AverageTemperature",
                            "rain_stdev", "Log_Yield"]

    assert len(fn.summary_stats(columns_to_summarize)) == 4

    assert len(fn.summary_stats(columns_to_summarize)[1]) == 13

    assert isinstance(fn.summary_stats(columns_to_summarize), list) is True

    columns_to_summarize = ["Total_Yield"]

    assert len(fn.summary_stats(columns_to_summarize)[0]) == 1

    assert len(fn.summary_stats(columns_to_summarize)) == 4

    print("summary stat tests passed")


def test_ols():
    """test whether ols and mse are calculated correctly by the
    ols_model function. Return statement will return fit model, but
    summary and mse will still be printed"""
    cleaned_data = fn.wrangle_frame()

    ols_model = fn.ols_model(cleaned_data)

    ols_fit = ols_model.fit()

    assert ols_fit.params["Intercept"] != 0

    assert len(ols_fit.params) == 13

    assert ols_fit.cov_type == "nonrobust"

    assert ols_fit.predict(
           exog=cleaned_data.quantile(.50))[0.5] > ols_fit.predict(
           exog=cleaned_data.quantile(.90))[0.90]

    assert len(ols_fit.predict()) == len(cleaned_data["Log_Yield"])
    print("ols tests passed")


def test_random_forest():
    """test whether random forest and
    mse are correctly and consistently calculated
    by ht erandom_forest_model function"""
    data = fn.wrangle_frame()

    def subtest_1():
        fn.random_forest_model(data)
        assert plt.gcf().number == 11

    def subtest_2():
        plt.figure(10)
        num_figures_before = plt.gcf().number
        fn.random_forest_model(data)
        num_figures_after = plt.gcf().number
        assert num_figures_before < num_figures_after

        subtest_1

        subtest_2

    assert isinstance(fn.random_forest_model(data), float) is True

    assert fn.random_forest_model(
           data) >= -1 and fn.random_forest_model(data) <= 0

    print("random forest test passed")


def test_main():
    """main function to run all tests"""

    test_rain_df()

    test_temp()

    test_yield()

    test_fert()

    test_gni()

    test_manure()

    test_pest()

    test_land()

    test_data_wrangle()

    test_time_plot()

    test_scatter_plots()

    test_summary_stats()

    test_ols()

    test_random_forest()


test_main()
