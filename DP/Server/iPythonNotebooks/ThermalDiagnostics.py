import sys
sys.path.append("..")
sys.path.append("../MPC")


import pandas as pd
import numpy as np
import pickle
import datetime
import yaml
import pytz
import pprint
import datetime
import utils






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
    no_action_data = data[data["action"] == utils.NO_ACTION]
    # TODO get it to work with 2 stage cooling stuff.
    heating_data = data[utils.is_heating(data["action"])]
    cooling_data = data[utils.is_cooling(data["action"])]

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
    print("--------- Find data where heating happened but temperature dropped even though all other "
          "zones/outside temperatures where higher ---------")
    row_filter = ["zone_temperature_" in col for col in data.columns]

    res = []

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

                res.append(row.name)

        elif others_less_than_curr_zone and tout < tin and action != 1 and action != 3:
            if tin < tnext:
                print("All other temperature were lower than curr temperature, but temperature rose:")
                print(row)
                print("")

                res.append(row.name)

    return res

    # ====== end ======


# ============ CONSISTENCY CHECKS ==========

def apply_consistency_check_to_model(data=None, thermal_model=None, thermal_model_class=None):
    """
    :param data: Only used for fitting the thermal model."""
    assert thermal_model is not None or (thermal_model_class is not None and data is not None)

    # evaluate actions. On same temperatures, heating should increase, cooling decrease, and no action should be no different
    if thermal_model is None:
        thermalModel = thermal_model_class()
        thermalModel.fit(data, data["t_next"])
    else:
        thermalModel = thermal_model

    def prepare_consistency_test_data(thermal_model, start_temperature=50, end_temperature=100, increments=5, dt=5):
        filter_columns = thermal_model._filter_columns
        data = []
        for temperature in range(start_temperature, end_temperature + increments, increments):
            # TODO potentially not hardcode dt
            for action in range(0, 6):
                datapoint = {"dt": dt, "action": action,
                             "t_out": temperature, "t_in": temperature}
                for col in filter_columns:
                    if "zone_temperature_" in col:
                        datapoint[col] = temperature
                data.append(datapoint)
        return pd.DataFrame(data)

    consistancy_test_data = prepare_consistency_test_data(thermalModel)
    consistancy_test_data["prediction"] = thermalModel.predict(consistancy_test_data, should_round=False)

    def consistency_check(df):
        """Consistency check for a df with 3 entries. The datapoints can only differ in the action to be meaningful.
        :param df: pd.df columns as given by ThermalDataManager plus a column with the predctions"""
        t_in = df['t_in'].values[0]
        dt = df["dt"].values[0]

        no_action_temperature = df[(df['action'] == utils.NO_ACTION)]["prediction"].values
        heating_temperature = df[df['action'] == utils.HEATING_ACTION]["prediction"].values
        cooling_temperature = df[df['action'] == utils.COOLING_ACTION]["prediction"].values
        two_stage_heating_temperature = df[df['action'] == utils.TWO_STAGE_HEATING_ACTION]["prediction"].values
        two_stage_cooling_temperature = df[df['action'] == utils.TWO_STAGE_COOLING_ACTION]["prediction"].values

        consistency_flag = True

        # TODO only use this check when t_out and zone temperature are the same as t_in
        # Following checks with t_in are only possible when everything has the same temperature
        # check if predicted heating temperature is higher than current
        if heating_temperature <= t_in:
            consistency_flag = False
            print("Warning, heating_temperature is lower than t_in.")
        if cooling_temperature >= t_in:
            consistency_flag = False
            print("Warning, cooling_temperature is higher than t_in.")

        # check that heating is more than no action and cooling
        if heating_temperature <= no_action_temperature or heating_temperature <= cooling_temperature:
            consistency_flag = False
            print("Warning, heating_temperature is too low compared to other actions.")

        # check that two stage heating is more than no action and cooling
        if heating_temperature <= no_action_temperature or heating_temperature <= cooling_temperature:
            consistency_flag = False
            print("Warning, heating_temperature is too low compared to other actions.")

        # check cooling is lower than heating and no action
        if cooling_temperature >= no_action_temperature or cooling_temperature >= heating_temperature:
            consistency_flag = False
            print("Warning, cooling_temperature is too high compared to other actions.")
        # check if no action is between cooling and heating

        if not cooling_temperature < no_action_temperature < heating_temperature:
            consistency_flag = False
            print("Warning, no action is not larger than heating temperature and lower than cooling temperature.")

        # want to know for what data it didn't work
        if not consistency_flag:
            print("Inconsistency for following data:")
            print(df)
            print("")
        return consistency_flag

    consistentcy_results = consistancy_test_data.groupby(["t_in", "dt"]).apply(lambda df: consistency_check(df))
    is_zone_consistent = all(consistentcy_results.values)
    if is_zone_consistent:
        print("The thermal model is consistent.")
    else:
        print("The thermal model is inconsistent.")

# ============ END CONSISTENCY CHECK ========

if __name__ == '__main__':
    bldg = 'avenal-veterans-hall'
    zone_thermal_data = utils.get_data(building=bldg, days_back=150, evaluate_preprocess=False, force_reload=False)


    zone, zone_data = zone_thermal_data.items()[5]
    # filter to only have 5 min data
    # zone_data = zone_data[zone_data["dt"] == 5]
    # zone_data = zone_data[zone_data["t_min"] != zone_data["t_max"]]
    # zone_data = zone_data[(zone_data["t_in"] > zone_data["t_next"]) & (zone_data["action"] == utils.COOLING_ACTION)]

    print("Evaluate zone %s" % zone)
    evaluate_zone_data(zone_data)
    print(zone_data.index)

    # print(res)
    # zone_data = zone_data.drop(res)

    # evaluate_zone_data(zone_data)


    print("Consistency check for zone %s" % zone)
    from ThermalModel import ThermalModel
    thermal_model = ThermalModel().fit(zone_data, zone_data["t_next"])
    apply_consistency_check_to_model(thermal_model=thermal_model)



