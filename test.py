"""tests for functions in module"""
import pandas as pd
import matplotlib.pyplot as plt
import functions as fn


wrangle = fn.wrangle_frame()


def test_rain_df():
    """test rain_df correct output"""
    rain_df = pd.read_csv("aquastat.csv")

    rain_clean = fn.rain_frame()

    assert len(rain_clean.columns) == 2

    assert len(rain_clean["rain_stdev"]) <= len(rain_df["Value"])

    assert len(rain_clean.loc[~(rain_clean == 0).all(axis=1)][
           "rain_stdev"]) <= len(rain_df["Value"])

    assert len(rain_clean.dropna()[
           "rain_stdev"]) == len(rain_clean["rain_stdev"])
    print("test_rain passed")


def test_temp():
    """test temp_df correct output"""
    temp_df = pd.read_csv("tempstat.csv")

    temp_clean = fn.temp_frame()

    assert len(temp_clean.columns) == 3

    assert len(temp_clean["AverageTemperature"]) <= len(temp_df[
                                                         "AverageTemperature"])

    assert len(temp_clean.dropna()[
           "AverageTemperature"]) == len(temp_clean["AverageTemperature"])
    print("test_temp passed")


def test_yield():
    """test yield_df correct output"""
    yield_df = pd.read_csv("FAOSTAT_data_croptype.csv")

    yield_clean = fn.yield_frame()

    assert len(yield_clean.columns) == 3

    assert len(yield_clean["Yield_hg_ha"]) <= len(yield_df["Value"])

    assert len(yield_clean.loc[~(yield_clean == 0).all(
                                 axis=1)]["Yield_hg_ha"]) == len(
                                yield_clean["Yield_hg_ha"])

    assert len(yield_clean.dropna()[
           "Yield_hg_ha"]) == len(yield_clean["Yield_hg_ha"])

    print("test yield passed")


def test_land():
    """test land_df correct output"""
    land_df = pd.read_csv("FAOSTAT_data_croparea.csv")

    land_clean = fn.land_frame()

    assert len(land_clean.columns) == 3

    assert len(land_clean["ha_cropland"]) <= len(land_df["Value"])

    assert len(land_clean.loc[~(land_clean == 0).all(
                              axis=1)]["ha_cropland"]) == len(
                              land_clean["ha_cropland"])

    assert len(land_clean.dropna()[
           "ha_cropland"]) == len(land_clean["ha_cropland"])

    print("test land passed")


def test_fert():
    """test_fert_df correct ouput"""

    fert_df = pd.read_csv("FAOSTAT_data_fertilizer.csv")

    fert_clean = fn.fert_frame()

    assert len(fert_clean.columns) == 5

    assert len(fert_clean["kg_nitrogen"]) <= len(fert_df["Value"])

    assert len(fert_clean.dropna()[
           "kg_nitrogen"]) == len(fert_clean["kg_nitrogen"])

    print("test land passed")


def test_manure():
    """test manure_df correct ouput"""
    manure_df = pd.read_csv("FAOSTAT_data_manure.csv")

    manure_clean = fn.manure_frame()

    assert len(manure_clean.columns) == 3

    assert len(manure_clean["kg_manure"]) <= len(manure_df["Value"])

    assert len(manure_clean.dropna()[
           "kg_manure"]) == len(manure_clean["kg_manure"])

    print("test manure passed")


def test_pest():
    """test pest_df correct ouput"""
    pest_df = pd.read_csv("FAOSTAT_data_pesticides.csv")

    pest_clean = fn.pest_frame()

    assert len(pest_clean.columns) == 5

    assert len(pest_clean["kg_Herbicides"]) <= len(pest_df["Value"])

    assert len(pest_clean.dropna()[
           "kg_Herbicides"]) == len(pest_clean["kg_Herbicides"])

    print("test pest passed")


def test_gni():
    """test gni_frame() for correct output"""
    gni_df = pd.read_csv("gni_data.csv")

    gni_clean = fn.gni_frame()

    assert len(gni_clean.columns) == 5

    assert len(gni_clean["GNI"]) >= len(gni_df["1990"])

    assert len(gni_clean.dropna()[
           "GNI"]) == len(gni_clean["GNI"])

    assert max(gni_clean["post_2008"]) == 1

    print("test gni passed")


def test_data_wrangle():
    """will test the data_wrangle function
    to verify 16 columns """

    df_yield = fn.yield_frame()

    assert len(wrangle.columns) == 17

    assert max(wrangle["Year"]) == 2013

    assert min(wrangle["Year"]) == 1990

    assert len(wrangle["Total_Yield"]) <= len(
               df_yield["Yield_hg_ha"])
    assert len(wrangle.dropna()["Total_Yield"]) == len(
               wrangle["Total_Yield"])
    print("data wrangle tests passed")


def test_fung_fert_time_plot():
    """will test the graph_agri function
    in order to check that graphs are returned without
    error. Preliminary plans to make 6 plots"""

    def subtest_1():
        plt.figure(1)
        num_figures_before = plt.gcf().number
        fn.fung_fert_time_plot(wrangle)
        num_figures_after = plt.gcf().number
        assert num_figures_before < num_figures_after

    def subtest_2():
        fn.fung_fert_time_plot(wrangle)
        assert plt.gcf().number == 2

    subtest_1()

    subtest_2()

    print("time plot tests passed")


def test_yield_time_plot():
    """will test the graph_agri function
    in order to check that graphs are returned without
    error. Preliminary plans to make 6 plots"""

    def subtest_1():
        plt.figure(2)
        num_figures_before = plt.gcf().number
        fn.yield_time_plot(wrangle)
        num_figures_after = plt.gcf().number
        assert num_figures_before < num_figures_after

    def subtest_2():
        fn.yield_time_plot(wrangle)
        assert plt.gcf().number == 5

    subtest_1()

    subtest_2()

    print("time plot tests passed")


def test_scatter_plots():
    """will test the graph_agri function
    in order to check that graphs are returned without
    error. Preliminary plans to make 6 plots"""
    def subtest_1():
        fn.scatter_plots(wrangle)
        assert plt.gcf().number == 10

    def subtest_2():
        plt.figure(5)
        num_figures_before = plt.gcf().number
        fn.scatter_plots(wrangle)
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

    sum_stat = fn.summary_stats(columns_to_summarize)

    assert len(sum_stat) == 4

    assert len(sum_stat[1]) == 13

    assert isinstance(sum_stat, list) is True

    columns_to_summarize = ["Total_Yield"]

    sum_stat = fn.summary_stats(columns_to_summarize)

    assert len(sum_stat[0]) == 1

    assert len(sum_stat) == 4

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

    model = fn.random_forest_model(data)

    def subtest_1():
        plt.figure(1)
        fn.random_forest_model(data)
        assert plt.gcf().number == 11

    def subtest_2():
        plt.figure(10)
        num_figures_before = plt.gcf().number
        fn.random_forest_model(data)
        num_figures_after = plt.gcf().number
        assert num_figures_before < num_figures_after

        subtest_1()

        subtest_2()

    assert isinstance(model, float) is True

    assert model >= -1

    assert model <= 0

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

    test_fung_fert_time_plot()

    test_yield_time_plot()

    test_scatter_plots()

    test_summary_stats()

    test_ols()

    test_random_forest()


test_main()
