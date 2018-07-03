import sys
sys.path.append("..")
sys.path.append("../MPC")

from DP.Server.ThermalDataManager import ThermalDataManager
import pandas as pd
import numpy as np
import pickle
import datetime
import yaml
import pytz
import pprint
import datetime



import matplotlib.pyplot as plt
from xbos import get_client

from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

# ========= USEFUL DATA FUNCTIONS ========

def get_raw_data(building=None, cfg=None, days_back=50, force_reload=False):
    """
    Get raw inside temperature and outside temperature data as returned by mdal.
    :param building: (str) building name
    :param cfg: (dictionary) config file for building
    :param days_back: how many days back from current moment.
    :param force_reload: (boolean) If some data for this building is stored, the reload if not force reload. Otherwise,
                        load data as specified.
    :return: inside_data {zone: pd.df (t_in, a)}, outside_data pd.df (t_out)
    """
    assert cfg is not None or building is not None
    if cfg is not None:
        building = cfg["Building"]
    else:
        config_path = "../Buildings/" + building + "/" + building + ".yml"
        try:
            with open(config_path, "r") as f:
                cfg = yaml.load(f)
        except:
            print("ERROR: No config file for building %s with path %s" % (building, config_path))
            return

    print("----- Get data for Building: %s -----" % building)

    path = "./Eval_data/" + building + "_raw"

    # TODO ugly try/except

    # inside and outside data data
    import pickle
    try:
        assert not force_reload
        with open(path + "_inside", "r") as f:
            inside_data = pickle.load(f)
        with open(path + "_outside", "r") as f:
            outside_data = pickle.load(f)
    except:
        c = get_client()
        dataManager = ThermalDataManager(cfg, c)
        inside_data = dataManager._get_inside_data(dataManager.now - datetime.timedelta(days=days_back),
                                                   dataManager.now)
        outside_data = dataManager._get_outside_data(dataManager.now - datetime.timedelta(days=days_back),
                                                     dataManager.now)
        with open(path + "_inside", "wb") as f:
            pickle.dump(inside_data, f)
        with open(path + "_outside", "wb") as f:
            pickle.dump(outside_data, f)
    return inside_data, outside_data


# Look at a specific file
def get_data(building=None, cfg=None, days_back=50, evaluate_preprocess=False, force_reload=False):
    """
    Get preprocessed data.
    :param building: (str) building name
    :param cfg: (dictionary) config file for building. If none, the method will try to find it. 
    :param days_back: how many days back from current moment.
    :param evaluate_preprocess: (Boolean) should controller data manager add more features to data.
    :param force_reload: (boolean) If some data for this building is stored, the reload if not force reload. Otherwise,
                        load data as specified.
    :return: {zone: pd.df with columns according to evaluate_preprocess}
    """
    assert cfg is not None or building is not None
    if cfg is not None:
        building = cfg["Building"]
    else:
        config_path = "../Buildings/" + building + "/" + building + ".yml"
        try:
            with open(config_path, "r") as f:
                cfg = yaml.load(f)
        except:
            print("ERROR: No config file for building %s with path %s" % (building, config_path))
            return

    print("----- Get data for Building: %s -----" % building)

    if evaluate_preprocess:
        path = "./Eval_data/" + building + "_eval"
    else:
        path = "./Eval_data/" + building

    # TODO ugly try/except
    try:
        assert not force_reload
        with open(path, "r") as f:
            import pickle
            thermal_data = pickle.load(f)
    except:
        c = get_client()
        dataManager = ThermalDataManager(cfg, c)
        thermal_data = dataManager.thermal_data(days_back=days_back, evaluate_preprocess=evaluate_preprocess)
        with open(path, "wb") as f:
            import pickle
            pickle.dump(thermal_data, f)
    return thermal_data


def concat_zone_data(thermal_data):
    """Concatinates all zone data into one big dataframe. Will sort by index. Get rid of all zone_temperature columns.
    :param thermal_data: {zone: pd.df}
    :return pd.df without zone_temperature columns"""
    concat_data = pd.concat(thermal_data.values()).sort_index()
    filter_columns = ["zone_temperature" not in col for col in concat_data.columns]
    return concat_data[concat_data.columns[filter_columns]]


def get_config(building):
    config_path = "../Buildings/" + building + "/" + building + ".yml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.load(f)
    except:
        print("ERROR: No config file for building %s with path %s" % (building, config_path))
        return
    return cfg

# ======== END DATA FUNCTIONS ========


def apply_temperature_change(df):
    no_change_data = df[df["t_next"] == df["t_in"]]
    increase_data = df[df["t_next"] > df["t_in"]]
    decrease_data = df[df["t_next"] < df["t_in"]]
    to_return = {"total": df.shape[0],
                 "total_no_change": no_change_data.shape[0],
                 "percent_no_change": 100 * no_change_data.shape[0] / float(df.shape[0]),
                 "total_increase": increase_data.shape[0],
                 "percent_increase": 100 * increase_data.shape[0] / float(df.shape[0]),
                 "total_decrease": decrease_data.shape[0],
                 "percent_decrease": 100 * decrease_data.shape[0] / float(df.shape[0])}
    return pd.Series(to_return)


def no_temperature_change_interval(df):
    def no_temp_change_for_action(a_df):
        no_change = a_df[a_df["t_min"] == a_df["t_max"]]
        return pd.Series({"total": a_df.shape[0], "percent_no_change": 100 * no_change.shape[0] / float(a_df.shape[0]),
                          "total_no_change": no_change.shape[0]})

    return df.groupby("action").apply(no_temp_change_for_action)


def evaluate_zone_data(data, find_single_actions=False):
    """Does basic data evaluation.
    :param data: (pd.df) data for one zone. columns: (t_in, t_next, t_out, a1, a2, zone_temperature+zones* )
    :param find_single_actions: (Boolean) whether to find actions which were less than or equal to a minute, but
        aren't preceeded or suceeded by the same action. Will only work with pure zone data."""
    # Filter for action data
    no_action_data = data[(data["a1"] == 0) & (data["a2"] == 0)]
    heating_data = data[data["a1"] == 1]
    cooling_data = data[data["a2"] == 1]

    # ====== Find the mean increase in temperature for each action for the given zone data ======
    print("----- Get avearge change in temperature for each action. -----")

    def get_delta_mean(action_data):
        # get the mean change of temperature from now to next.
        return np.mean((action_data["t_next"] - action_data["t_in"]) / action_data["dt"])

    mean_cooling_delta = get_delta_mean(cooling_data)
    mean_heating_delta = get_delta_mean(heating_data)
    mean_no_action_delta = get_delta_mean(no_action_data)
    mean_all_action_delta = get_delta_mean(data)

    print("For cooling there was an average %s degree change." % str(mean_cooling_delta))
    print("For heating there was an average %s degree change." % str(mean_heating_delta))
    print("For no action there was an average %s degree change." % str(mean_no_action_delta))
    print("For all actions there was an average %s degree change." % str(mean_all_action_delta))

    # ====== end ======

    # ====== Number and percentage of individual action data ========
    print("--------- Number and percentage of individual action data ---------")
    num_data = data.shape[0]
    num_no_action_data = no_action_data.shape[0]
    num_heating_data = heating_data.shape[0]
    num_cooling_data = cooling_data.shape[0]

    print("We have %f total data points." % num_data)

    print("We have %f no action data points, which is %f percent of total." % (
    num_no_action_data, 100 * float(num_no_action_data) / num_data))

    print("We have %f heating data points, which is %f percent of total." % (
    num_heating_data, 100 * float(num_heating_data) / num_data))

    print("We have %f cooling data points, which is %f percent of total." % (
    num_cooling_data, 100 * float(num_cooling_data) / num_data))
    # ========= end ==========


    # ========= Find temperature change by action ========
    print("--------- Find temperature change by action ---------")

    no_action_change = apply_temperature_change(no_action_data)
    heating_change = apply_temperature_change(heating_data)
    cooling_change = apply_temperature_change(cooling_data)

    print("No action temperature changes:")
    print(no_action_change)
    print("")

    print("Heating temperature changes:")
    print(heating_change)
    print("")

    print("Cooling temperature changes:")
    print(cooling_change)
    print("")

    # ======== end =======

    # Group all data by dt and find evaluate temperature changes.
    print("---------- Group all data by dt and find evaluate temperature changes. --------")

    def group_dt_and_action(to_group_data):
        """Group the data by the interval lengths and whether temperature increased, dropped or stayed the same."""
        return to_group_data.groupby(by=["dt"]).apply(apply_temperature_change)

    no_action_dt_change = group_dt_and_action(no_action_data)
    heating_dt_change = group_dt_and_action(heating_data)
    cooling_dt_chage = group_dt_and_action(cooling_data)

    print("No Action, Action Change data:")
    print(no_action_dt_change)
    print("")

    print("Heating, Action Change data:")
    print(heating_dt_change)
    print("")

    print("Cooling, Action Change data:")
    print(cooling_dt_chage)
    print("")

    # ====== end =======

    if find_single_actions:
        # ====== Find actions which are not integers and stand by themselves
        print("--------- Find non interger actions with no action pre/suceeding ---------")
        idx_action = []
        for i in range(1, len(data.index) - 1):
            curr = data["action"][i]
            prev = data["action"][i - 1]
            next_a = data["action"][i + 1]
            if curr not in [0, 1, 2] and prev == 0 and next_a == 0:
                print("++++++++++++")
                print("Action is by itself with action %f for data:" % curr)
                print(data.iloc[i - 2:i + 2])
                idx_action.append(i)
        print("-------------")
        print("There were %f lone standing actions out of %f data points." % (len(idx_action), data.shape[0]))

    # ====== end =======

    # ======= Find intervals where there was no change in temperature throughout the interval =======
    print("--------- Find data where temperature did not change in interval ---------")

    if "t_min" in data.columns:
        print(data.groupby("dt").apply(no_temperature_change_interval))
    else:
        print("We don't have evaluation type data. i.e. no [t_min, t_max, etc] fields.")


    # ======= end =====


    # ======= Find data where heating happened but temperature dropped even though all other zones/outside temperatures
    # where higher ===========
    print("======= Find data where heating happened but temperature dropped even though all other "
          "zones/outside temperatures where higher ===========")
    row_filter = ["zone_temperature_" in col for col in data.columns]
    for i in range(data.shape[0]):
        row = data.iloc[i]
        tout = row["t_out"]
        tin = row["t_in"]
        tnext = row["t_next"]
        zone_temperatures = row[row_filter]
        others_less_than_curr_zone = all([temp < tin for temp in zone_temperatures])
        others_higher_than_curr_zone = all([temp > tin for temp in zone_temperatures])
        action = row["action"]

        if action == 0:
            continue

        if others_higher_than_curr_zone and tout > tin and action != 2 and action != 5:
            if tin > tnext:
                print("All other temperature were higher than curr temperature, but temperature fell:")
                print(row)
                print("")
        elif others_less_than_curr_zone and tout < tin and action != 1 and action != 3:
            if tin < tnext:
                print("All other temperature were lower than curr temperature, but temperature rose:")
                print(row)
                print("")

    # ====== end ======


if __name__ == '__main__':
    bldg = 'avenal-veterans-hall'
    zone_thermal_data = get_data(building=bldg, days_back=50, evaluate_preprocess=True, force_reload=False)

    zone, zone_data = zone_thermal_data.items()[0]
    print("Evaluate zone %s" % zone)
    evaluate_zone_data(zone_data)




