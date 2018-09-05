# This is a file which provides either functions which simplify use or provide better abstraction in case
# something changes in config files etc.


import datetime
import math
import string
import sys

import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
import pytz
import yaml
from xbos import get_client
from xbos.devices.thermostat import Thermostat

import DataManager
# be careful of circular import.
# https://stackoverflow.com/questions/11698530/two-python-modules-require-each-others-contents-can-that-work
import ThermalDataManager

SERVER_DIR_PATH = UTILS_FILE_PATH = os.path.dirname(__file__)  # this is true for now

sys.path.append(SERVER_DIR_PATH + "/Lights")
import lights

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
# print("using package pygraphviz")
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
        # print("using package pydotplus")
    except ImportError:
        print()
        print("Both pygraphviz and pydotplus were not found ")
        print("see http://networkx.github.io/documentation"
              "/latest/reference/drawing.html for info")
        print()
        raise

'''
Utility constants
'''
NO_ACTION = 0
HEATING_ACTION = 1
COOLING_ACTION = 2
FAN = 3
TWO_STAGE_HEATING_ACTION = 4
TWO_STAGE_COOLING_ACTION = 5

SERVER_DIR_PATH = UTILS_FILE_PATH = os.path.dirname(__file__)  # this is true for now

'''
Utility functions
'''


# ============ BUILDING AND ZONE GETTER ========
def choose_building_and_zone():
    print "-----------------------------------"
    print "Buildings:"
    print "-----------------------------------"
    root, dirs, files = os.walk(SERVER_DIR_PATH + "/Buildings/").next()
    for index, building in enumerate(dirs, start=1):
        print index, building
    print "-----------------------------------"
    index = input("Please choose a building (give a number):") - 1
    building = dirs[index]
    print "-----------------------------------"
    print ""
    print "-----------------------------------"
    print "	" + str(building)
    print "-----------------------------------"
    print "-----------------------------------"
    print "Zones:"
    print "-----------------------------------"
    root, dirs, files = os.walk("../Buildings/" + str(building) + "/ZoneConfigs").next()
    for index, zones in enumerate(files, start=1):
        print index, zones[:-4]
    print "-----------------------------------"
    index = input("Please choose a zone (give a number):") - 1
    zone = files[index][:-4]
    print "-----------------------------------"
    print "-----------------------------------"
    print "	" + str(building)
    print "	" + str(zone)
    print "-----------------------------------"
    return building, zone


# ============ DATE FUNCTIONS ============

def get_utc_now():
    """Gets current time in utc time.
    :return Datetime in utctime zone"""
    return datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))


def in_between(now, start, end):
    """Finds whether now is between start and end. Takes care of cases such as start=11:00pm and end=1:00am 
    now = 00:01, and hence would return True. 
    :param now: (datetime.time) 
    :param start: (datetime.time) 
    :param end: (datetime.time) 
    :return (boolean)"""
    if start < end:
        return start <= now < end
    # when end is in the next day.
    elif end < start:
        return start <= now or now < end
    else:
        return True


def combine_date_time(time, date):
    """Combines the time and date to a combined datetime. Specific use in DataManager functions.
    :param time: (str) HH:MM
    :param date: (datetime)
    :returns datetime with date from date and time from time. But with seconds as 0."""
    datetime_time = get_time_datetime(time)
    return date.replace(hour=datetime_time.hour, minute=datetime_time.minute, second=0, microsecond=0)
    # return datetime.datetime.combine(date, datetime_time)


def in_between_datetime(now, start, end):
    """Finds whether now is between start and end.
    :param now: (datetime) 
    :param start: (datetime) 
    :param end: (datetime) 
    :return (boolean)"""
    return start <= now <= end


def get_time_datetime(time_string):
    """Gets datetime from string with format HH:MM.
    :param date_string: string of format HH:MM
    :returns datetime.time() object with no associated timzone. """
    return datetime.datetime.strptime(time_string, "%H:%M").time()


def get_mdal_string_to_datetime(date_string, with_utc=True):
    """Gets datetime from string with format Year-Month-Day Hour:Minute:Second UTC. Note, string should be for utc
    time.
    :param date_string: string of format Year-Month-Day Hour:Minute:Second UTC.
    :param with_utc: boolean indicating wether to localize to UTC time.
    :returns datetime.time() object in UTC time or naive time. """
    date_datetime = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S %Z")
    if with_utc:
        return date_datetime.replace(tzinfo=pytz.timezone("UTC"))
    else:
        return date_datetime


def get_mdal_datetime_to_string(date_object):
    """Gets string from datetime object. In UTC Time.
    :param date_object
    :returns '%Y-%m-%d %H:%M:%S UTC' """
    return date_object.strftime('%Y-%m-%d %H:%M:%S') + ' UTC'


def get_datetime_to_string(date_object):
    """Gets string from datetime object.
    :param date_object
    :returns '%Y-%m-%d %H:%M:%S' """
    return date_object.strftime('%Y-%m-%d %H:%M:%S')


# ============ DATA FUNCTIONS ============


def prediction_test( X, thermal_model, is_two_stage):
    """Takes the data X and for each datapoint runs the thermal model for each possible actions and 
    sees if the prediction we would get is consistent and sensible. i.e. temperature change for heating is more than for 
    cooling etc. 
    Note, relies on having set self.is_two_stage to know if we are predicting two stage cooling data. """

    # get predictions for every action possible.
    def predict_action(X, action, thermal_model):
        unit_actions = np.ones(X.shape[0])
        X_copy = X.copy()
        X_copy["action"] = unit_actions * action
        return thermal_model.predict(X_copy)

    def consistency_check(row, is_two_stage):
        """check that the temperatures are in the order we expect them. Heating has to be strictly more than no
        action and cooling and cooling has to be strictly less than heating and no action.
        If we only want to check consistency for the doing nothing action, then it should suffice to only check
        that the max of cooling is smaller and the min of heating is larger. 
        :param row: pd.df keys which are the actions in all caps and "MAIN_ACTION" ,the action we actually 
                want to find the consistency for.
        :param is_two_stage: bool wether we are using predictions for a two stage building.
        :return bool. Whether the given data is consistent."""
        main_action = row["MAIN_ACTION"]

        if is_two_stage:
            # can't have cooling that's higher than 0 and heating that's lower.
            if max(row["COOLING_ACTION"], row["TWO_STAGE_COOLING_ACTION"]) >= row["T_IN"] or \
                            min(row["HEATING_ACTION"], row["TWO_STAGE_HEATING_ACTION"]) <= row["T_IN"]:
                return False

            # checks if the actions are in the correct order
            if main_action == NO_ACTION:
                if max(row["COOLING_ACTION"], row["TWO_STAGE_COOLING_ACTION"]) \
                        < row["NO_ACTION"] \
                        < min(row["HEATING_ACTION"], row["TWO_STAGE_HEATING_ACTION"]):
                    return True
            else:
                # TODO Maybe have only strictly greater.
                if row["TWO_STAGE_COOLING_ACTION"] <= row["COOLING_ACTION"] \
                        < row["NO_ACTION"] \
                        < row["HEATING_ACTION"] <= row["TWO_STAGE_HEATING_ACTION"]:
                    return True
        else:
            # can't have cooling that's higher than 0 and heating that's lower.
            if row["COOLING_ACTION"] >= row["T_IN"] or \
                            row["HEATING_ACTION"] <= row["T_IN"]:
                return False

            # checks if the actions are in the correct order
            if row["COOLING_ACTION"] < row["NO_ACTION"] < row["HEATING_ACTION"]:
                return True

        return False

    def sensibility_check(X, is_two_stage, sensibility_measure=20):
        """Checks if the predictions are within sensibility_measure degrees of tin. It wouldn't make sense to predict 
        more. This will be done for all possible action predictions. If any one is not sensible, we will disregard 
        the whole prediction set. 
        ALSO, check if Nan values
        :param X: pd.df keys which are the actions in all caps and "MAIN_ACTION" ,the action we actually 
                want to find the consistency for.
        :param is_two_stage: bool wether we are using predictions for a two stage building.
        :param sensibility_measure: (Float) degrees within the prediction may lie. 
                        e.g. t_in-sensibility_measure < prediction < t_in+sensibility_measure
        :return np.array booleans"""
        differences = []
        differences.append(np.abs(X["NO_ACTION"] - X["T_IN"]))
        differences.append(np.abs(X["HEATING_ACTION"] - X["T_IN"]))
        differences.append(np.abs(X["COOLING_ACTION"] - X["T_IN"]))
        if is_two_stage:
            differences.append(np.abs(X["TWO_STAGE_HEATING_ACTION"] - X["T_IN"]))
            differences.append(np.abs(X["TWO_STAGE_COOLING_ACTION"] - X["T_IN"]))


        # check if every difference is in sensible band and not nan. We can check if the prediction is nan
        # by checking if the difference is nan, because np.nan + x = np.nan
        sensibility_filter_array = [(diff < sensibility_measure) & (diff != np.nan) for diff in differences]
        # putting all filters together by taking the and of all of them.
        sensibility_filter_check = reduce(lambda x, y: x & y, sensibility_filter_array)
        return sensibility_filter_check.values


    no_action_predictions = predict_action(X, NO_ACTION, thermal_model)
    heating_action_predictions = predict_action(X, HEATING_ACTION, thermal_model)
    cooling_action_predictions = predict_action(X,  COOLING_ACTION, thermal_model)
    if is_two_stage:
        two_stage_heating_predictions = predict_action(X,  TWO_STAGE_HEATING_ACTION, thermal_model)
        two_stage_cooling_predictions = predict_action(X,  TWO_STAGE_COOLING_ACTION, thermal_model)
    else:
        two_stage_cooling_predictions = None
        two_stage_heating_predictions = None

    predictions_action = pd.DataFrame({"T_IN": X["t_in"], "NO_ACTION": no_action_predictions,
                                       "HEATING_ACTION": heating_action_predictions,
                                       "COOLING_ACTION": cooling_action_predictions,
                                       "TWO_STAGE_HEATING_ACTION": two_stage_heating_predictions,
                                       "TWO_STAGE_COOLING_ACTION": two_stage_cooling_predictions,
                                       "MAIN_ACTION": X["action"]})

    consistent_filter = predictions_action.apply(lambda row: consistency_check(row, is_two_stage), axis=1)
    sensibility_filter = sensibility_check(predictions_action, is_two_stage, sensibility_measure=20)

    return consistent_filter.values & sensibility_filter

def and_dictionary(a_dict, b_dict):
    """
    Takes the and of two boolean dictionaries with same keys.
    :param a_dict: boolean dictionary
    :param b_dict: boolean dictionary
    :return: dictionary of length a_dict with the and of both dictionaries
    """
    assert a_dict.keys() == b_dict.keys()
    return {iter_zone: a_dict[iter_zone] and b_dict[iter_zone] for iter_zone in a_dict.keys()}


def round_increment(data, precision=0.05):
    """Round to nearest increment of precision.
    :param data: np.array of floats or single float
    :param precision: (float) the increment to round to
    :return (np.array or float) of rounded floats."""
    # source for rounding: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    return precision * np.round(data / float(precision))


def is_cooling(action_data):
    """Returns boolen array of actions which were cooling (either two or single stage).
    :param action_data: np.array or pd.series"""
    return (action_data == COOLING_ACTION) | (action_data == TWO_STAGE_COOLING_ACTION)


def is_heating(action_data):
    """Returns boolen array of actions which were heating (either two or single stage).
    :param action_data: np.array or pd.series"""
    return (action_data == HEATING_ACTION) | (action_data == TWO_STAGE_HEATING_ACTION)


def is_DR(now, cfg_building):
    """
    Whether we are in a DR timeperiod while having the DR flag set.
    :param now: datetime timezone aware.
    :param cfg_building: The config file for the building.
    :return: Bool. Whether in DR region.
    """
    now_cfg_timezone = now.astimezone(tz=get_config_timezone(cfg_building))

    dr_start = cfg_building["Pricing"]["DR_Start"]
    dr_end = cfg_building["Pricing"]["DR_Finish"]
    cfg_now_time = now_cfg_timezone.time()
    is_inbetween = in_between(now=cfg_now_time, start=get_time_datetime(dr_start),
                              end=get_time_datetime(dr_end))
    is_dr_day = cfg_building["Pricing"]["DR"]
    return is_dr_day and is_inbetween


def get_config_timezone(cfg_building):
    """
    Gets the timezone object for the building.
    :param cfg_building: The config file of the whole building.
    :return: Pytz object. 
    """
    return pytz.timezone(cfg_building["Pytz_Timezone"])


def choose_client(cfg=None):
    if cfg is not None and cfg["Server"]:
        client = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        client = get_client()
    return client


def get_config(building):
    config_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + building + ".yml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.load(f)
    except:
        raise Exception("ERROR: No config file for building %s with path %s" % (building, config_path))
    return cfg


def get_zone_config(building, zone):
    config_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + "ZoneConfigs/" + zone + ".yml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.load(f)
    except:
        raise Exception(
            "ERROR: No config file for building %s and zone % s with path %s" % (building, zone, config_path))
    return cfg


def get_zone_log(building, zone):
    # TODO Fix this function.
    log_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + "Logs/" + zone + ".log"

    ## fix for one lines
    try:

        f = open(log_path, "r")
        log = f.read()
        log = string.replace(log, "UTCTHERMOSTAT", "UTC\nTHERMOSTAT")
        f.close()

        f = open(log_path, 'w')
        f.write(log)
        f.close()
    except:
        print("ERROR: No config file for building %s and zone % s with path %s" % (building, zone, log_path))
        return
    ## end of fix DELETE THIS WHEN ALL LOGS ARE FIXED!

    try:
        with open(log_path, "r") as f:
            ### fix for same line logs ###
            log = f.readlines()
    except:
        print("ERROR: No config file for building %s and zone % s with path %s" % (building, zone, log_path))
        return
    return log


# Maybe put in ThermalDataManager because of circular import.
def get_data(building=None, client=None, cfg=None, start=None, end=None, days_back=50, evaluate_preprocess=False,
             force_reload=False):
    """
    Get preprocessed data.
    :param building: (str) building name
    :param cfg: (dictionary) config file for building. If none, the method will try to find it. 
    :param days_back: how many days back from current moment.
    :param evaluate_preprocess: (Boolean) should controller data manager add more features to data.
    :param force_reload: (boolean) If some data for this building is stored, the reload if not force reload. Otherwise,
                        load data as specified.
    :param start: the start time for the data. If none is given, we will use days_back to go back from the
                     end datetime if given (end - days_back), or the current time.
    :param end: the end time for the data. If given, we will use it as our end. If not given, we will use the current 
                    time as the end.
    :return: {zone: pd.df with columns according to evaluate_preprocess}
    """
    assert cfg is not None or building is not None
    if cfg is not None:
        building = cfg["Building"]
    else:
        cfg = get_config(building)

    print("----- Get data for Building: %s -----" % building)

    if evaluate_preprocess:
        path = SERVER_DIR_PATH + "/Thermal_Data/" + building + "_eval"
    else:
        path = SERVER_DIR_PATH + "/Thermal_Data/" + building

    if end is None:
        end = get_utc_now()
    if start is None:
        start = end - datetime.timedelta(days=days_back)

    # TODO ugly try/except
    try:
        assert not force_reload
        print(path)
        with open(path, "r") as f:
            import pickle
            thermal_data = pickle.load(f)
    except:
        if client is None:
            client = choose_client(cfg)
        dataManager = ThermalDataManager.ThermalDataManager(cfg, client)
        thermal_data = dataManager.thermal_data(start=start, end=end, evaluate_preprocess=evaluate_preprocess)
        with open(path, "wb") as f:
            import pickle
            pickle.dump(thermal_data, f)
    return thermal_data


def get_raw_data(building=None, client=None, cfg=None, start=None, end=None, days_back=50, force_reload=False):
    assert cfg is not None or building is not None
    if cfg is not None:
        building = cfg["Building"]
    else:
        config_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + building + ".yml"
        try:
            with open(config_path, "r") as f:
                cfg = yaml.load(f)
        except:
            print("ERROR: No config file for building %s with path %s" % (building, config_path))
            return

    print("----- Get data for Building: %s -----" % building)

    path = SERVER_DIR_PATH + "/Thermal_Data/" + building
    # TODO ugly try/except

    if end is None:
        end = get_utc_now()
    if start is None:
        start = end - datetime.timedelta(days=days_back)

    # inside and outside data data
    import pickle
    try:
        assert not force_reload
        with open(path + "_inside", "r") as f:
            inside_data = pickle.load(f)
        with open(path + "_outside", "r") as f:
            outside_data = pickle.load(f)
    except:
        if client is None:
            client = get_client()
        dataManager = ThermalDataManager.ThermalDataManager(cfg, client)

        inside_data = dataManager._get_inside_data(start, end)
        outside_data = dataManager._get_outside_data(start, end)
        with open(path + "_inside", "wb") as f:
            pickle.dump(inside_data, f)
        with open(path + "_outside", "wb") as f:
            pickle.dump(outside_data, f)
    return inside_data, outside_data


def get_mdal_data(mdal_client, query):
    """Gets mdal data. Necessary method because if a too long time frame is queried, mdal does not return the data.
    :param mdal_client: mdal object to query data.
    :param query: mdal query
    :return pd.df with composition as columns. Timeseries in UTC time."""
    start = get_mdal_string_to_datetime(query["Time"]["T0"])
    end = get_mdal_string_to_datetime(query["Time"]["T1"])
    time_frame = end - start

    # get windowsize
    str_window = query["Time"]["WindowSize"]
    assert str_window[-3:] == "min"
    WINDOW_SIZE = datetime.timedelta(minutes=int(str_window[:-3]))

    if time_frame < WINDOW_SIZE:
        raise Exception("WindowSize is less than the time interval for which data is requested.")

    # To get logarithmic runtime we take splits which are powers of two.
    max_interval = datetime.timedelta(hours=12)  # the maximum interval length in which to split the data.
    max_num_splits = int(time_frame.total_seconds() // max_interval.total_seconds())
    all_splits = [1]
    for _ in range(2, max_num_splits):
        power_split = all_splits[-1] * 2
        if power_split > max_num_splits:
            break
        all_splits.append(power_split)

    received_all_data = False
    outside_data = []
    # start loop to get data in time intervals of logarithmically decreasing size. This will hopefully find the
    # spot at which mdal returns data.
    for num_splits in all_splits:
        outside_data = []
        pre_look_ahead = time_frame / num_splits

        # to round down to nearest window size multiple
        num_window_in_pre_look = pre_look_ahead.total_seconds() // WINDOW_SIZE.total_seconds()
        look_ahead = datetime.timedelta(seconds=WINDOW_SIZE.total_seconds() * num_window_in_pre_look)

        print("Attempting to get data in %f day intervals." % (look_ahead.total_seconds() / (60 * 60 * 24)))

        temp_start = start
        temp_end = temp_start + look_ahead

        while temp_end <= end:
            query["Time"]["T0"] = get_mdal_datetime_to_string(temp_start)
            query["Time"]["T1"] = get_mdal_datetime_to_string(temp_end)
            mdal_outside_data = mdal_client.do_query(query, tz="UTC")
            if mdal_outside_data == {}:
                print("Attempt failed.")
                received_all_data = False
                break
            else:
                outside_data.append(mdal_outside_data["df"])

                # advance temp_start and temp_end
                temp_start = temp_end + WINDOW_SIZE
                temp_end = temp_start + look_ahead

                # to get rest of data if look_ahead is not exact mutliple of time_between
                if temp_start < end < temp_end:
                    temp_end = end

                # To know that we received all data.
                if end < temp_start:
                    received_all_data = True

        # stop if we got the data
        if received_all_data:
            print("Succeeded.")
            break

    if not received_all_data:
        raise Exception("WARNING: Unable to get data form MDAL.")

    return pd.concat(outside_data)


def concat_zone_data(thermal_data):
    """Concatinates all thermal data zone data into one big dataframe. Will sort by index. Get rid of all zone_temperature columns.
    :param thermal_data: {zone: pd.df}
    :return pd.df without zone_temperature columns"""
    concat_data = pd.concat(thermal_data.values()).sort_index()
    filter_columns = ["zone_temperature" not in col for col in concat_data.columns]
    return concat_data[concat_data.columns[filter_columns]]


def as_pandas(result):
    time = result[list(result.keys())[0]][:, 0]
    df = pd.DataFrame(time, columns=['Time'])
    df['Time'] = pd.to_datetime(df['Time'], unit='s')

    for key in result:
        df[key] = result[key][:, 1].tolist()
        try:
            df[key + " Var"] = result[key][:, 2].tolist()
        except IndexError:
            pass

    df = df.set_index('Time')
    return df


def interpolate_to_start_end_range(data, start, end, interval):
    """
    Returns data but indexed with start to end in frequency of interval minutes. Interpolate the data when
    needed. Interpolating should give a fine approximation. However, while the new intervals represent the mean 
    temperatures of the time interval given, the mean of them throughout an hour will not necessarily be the
    mean from which we started from. 
    :param data: pd.series/pd.df has to have time series index which can contain a span from start to end.
    :param start: the start of the data we want
    :param end: the end of the data we want
    :param interval: minutes for the interval
    :return: data but with index of pd.date_range(start, end, interval)
    """
    # https://stackoverflow.com/questions/40034040/pandas-timeseries-resampling-and-interpolating-together
    date_range = pd.date_range(start, end, freq=str(interval) + "T")
    return data.reindex(date_range.union(data.index)).interpolate().loc[date_range]


def get_outside_temperatures(building_config, start, end, data_manager, thermal_data_manager, interval=15):
    """
    Get outside weather from start to end. Will combine historic and weather predictions data when necessary.
    :param start: datetime timezone aware
    :param end: datetime timezone aware
    :param interval: (int minutes) interval of the returned timeseries in minutes. 
    :return: pd.Series with combined data of historic and prediction outside weather. 
    """
    # we might have that the given now is before the actual current time
    # hence need to get historic data and combine with weather predictions.

    # For finding out if start or/and end are before or after the current time.
    utc_now = get_utc_now()

    # Set start and end to correct timezones
    cfg_timezone = get_config_timezone(building_config)
    start_utc = start.astimezone(tz=pytz.utc)
    end_utc = end.astimezone(tz=pytz.utc)
    start_cfg_timezone = start.astimezone(tz=cfg_timezone)
    end_cfg_timezone = end.astimezone(tz=cfg_timezone)
    now_cfg_timezone = utc_now.astimezone(tz=cfg_timezone)

    # If simulation window is partially in the past and in the future
    if in_between_datetime(utc_now, start_utc, end_utc):
        historic_start_utc = start_utc
        historic_end_utc = utc_now
        future_start_utc = utc_now
        future_end_utc = end_utc

        historic_start_cfg_timezone = start_cfg_timezone
        historic_end_cfg_timezone = now_cfg_timezone
        future_start_cfg_timezone = now_cfg_timezone
        future_end_cfg_timezone = end_cfg_timezone

    # If simulation window is fully in the future
    elif start_utc >= utc_now:
        historic_start_utc = None
        historic_end_utc = None
        future_start_utc = start_utc
        future_end_utc = end_utc

        historic_start_cfg_timezone = None
        historic_end_cfg_timezone = None
        future_start_cfg_timezone = start_cfg_timezone
        future_end_cfg_timezone = end_cfg_timezone

    # If simulation window is fully in the past
    else:
        historic_start_utc = start_utc
        historic_end_utc = end_utc
        future_start_utc = None
        future_end_utc = None

        historic_start_cfg_timezone = start_cfg_timezone
        historic_end_cfg_timezone = end_cfg_timezone
        future_start_cfg_timezone = None
        future_end_cfg_timezone = None

    # Populating the outside_temperatures pd.Series for MPC use. Ouput is in cfg timezone.
    outside_temperatures = pd.Series(index=pd.date_range(start_cfg_timezone, end_cfg_timezone, freq=str(interval)+"T"))
    if future_start_cfg_timezone is not None:
        future_weather = data_manager.weather_fetch(start=future_start_cfg_timezone, end=future_end_cfg_timezone, interval=interval)
        outside_temperatures[future_start_cfg_timezone:future_end_cfg_timezone] = future_weather.values

    # Combining historic data with outside_temperatures correctly if exists.
    if historic_start_cfg_timezone is not None:
        historic_weather = thermal_data_manager._get_outside_data(historic_start_utc,
                                                                  historic_end_utc, inclusive=True)
        historic_weather = thermal_data_manager._preprocess_outside_data(historic_weather.values()).squeeze()


        # Convert historic_weather to cfg timezone.
        historic_weather.index = historic_weather.index.tz_convert(tz=building_config["Pytz_Timezone"])

        # Make sure historic weather has correct interval and start to end times.
        historic_weather = interpolate_to_start_end_range(historic_weather,
                                                          historic_start_cfg_timezone, historic_end_cfg_timezone,
                                                          interval)

        # Populate outside data
        outside_temperatures[historic_start_cfg_timezone:historic_end_cfg_timezone] = historic_weather.values

    return outside_temperatures


def get_zones(building):
    """
    Gets all zone names for a building.
    Assumes that the zone config files end with .yml .
    :param building: (str) building name as in Buildings folder.
    :return: (str arr) the zone names. 
    """
    all_zones = []
    root, dirs, files = os.walk(SERVER_DIR_PATH + "/Buildings/" + str(building) + "/ZoneConfigs").next()
    for index, zone in enumerate(files, start=1):
        all_zones.append(zone[:-4])
    return all_zones


def get_zone_data_managers(building, zones, now, client):
    """
    Gets the data managers for each zone as a dictionary.
    :param building: (str) Building name 
    :param zones: (list str) The zones for which to get the data
    :param now: (datetime) the time for which the datamanager should return data.
    :return: {zone: DataManager}
    """
    # make now utc for datamanager
    if now is not None:
        utc_now = now.astimezone(tz=pytz.utc)
    else:
        utc_now = now

    building_cfg = get_config(building)

    zone_managers = {}
    for zone in zones:
        zone_config = get_zone_config(building, zone)
        zone_managers[zone] = DataManager.DataManager(controller_cfg=building_cfg, advise_cfg=zone_config,
                                                      client=client,
                                                      zone=zone, now=utc_now)
    return zone_managers


def get_occupancy_matrix(building, zones, start, end, interval, datamanager_zones=None):
    """
    Gets the occupancy matrix for the given building. This means that we get a dictionary index by zone
    where we get the occupancy for the given zone from start to end provided in the given interval. Taking the mean
    occupancy for the interval.
    :param building: (string) buidling name 
    
    :param zones: (list str) The zones for which to get the data.:param start_utc: (datetime) the start of the occupancy data. timezone aware.
    :param end_utc: (datetime) the end of the occupancy data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :param datamanager_zones: None or {zone: Datamanager} If not provided it will create new ones but will add another
                                second of runtime.
    :return: {zone: pd.df columns="occ" with timeseries data from start to end (inclusive) with the interval as 
     frequency. will be in timezone of config file} The intervals will be found by taking the mean occupancy in the 
     interval. 
    """
    # TODO implement archiver zone occupancy with predictions.

    return get_data_matrix(building, zones, start, end, interval, "occupancy", datamanager_zones)


def get_comfortband_matrix(building, zones, start, end, interval, datamanager_zones=None):
    """
    Gets the comfortband matrix for the given building. This means that we get a dictionary index by zone
    where we get the comfortband for the given zone from start to end provided in the given interval. Taking the mean
    comfortband for the interval.
    :param building: (string) buidling name 
    
    :param zones: (list str) The zones for which to get the data.:param start_utc: (datetime) the start of the comfortband data. timezone aware.
    :param end_utc: (datetime) the end of the comfortband data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :param datamanager_zones: None or {zone: Datamanager} If not provided it will create new ones but will add another
                                second of runtime.
    :return: {zone: pd.df columns="t_high", "t_low" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    """
    return get_data_matrix(building, zones, start, end, interval, "comfortband", datamanager_zones)


def get_safety_matrix(building, zones, start, end, interval, datamanager_zones=None):
    """
    Gets the safety matrix for the given building. This means that we get a dictionary index by zone
    where we get the safety for the given zone from start to end provided in the given interval. Taking the mean
    safety for the interval.
    :param building: (string) buidling name 
    :param zones: (list str) The zones for which to get the data.
    :param start_utc: (datetime) the start of the safety data. timezone aware.
    :param end_utc: (datetime) the end of the safety data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :param datamanager_zones: None or {zone: Datamanager} If not provided it will create new ones but will add another
                                second of runtime.
    :return: {zone: pd.df columns="t_high", "t_low" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    """
    return get_data_matrix(building, zones, start, end, interval, "safety", datamanager_zones)


def get_price_matrix(building, zones, start, end, interval, datamanager_zones=None):
    """
    Gets the price matrix for the given building. This means that we get a dictionary index by zone
    where we get the price for the given zone from start to end provided in the given interval. Taking the mean
    price for the interval.
    :param building: (string) building name 
    :param zones: (list str) The zones for which to get the data.
    :param start_utc: (datetime) the start of the data. timezone aware.
    :param end_utc: (datetime) the end of the data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :param datamanager_zones: None or {zone: Datamanager} If not provided it will create new ones but will add another
                                second of runtime.
    :return: {zone: pd.df columns="price" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    """
    return get_data_matrix(building, zones, start, end, interval, "prices", datamanager_zones)


def get_lambda_matrix(building, zones, start, end, interval, datamanager_zones=None):
    """
    Gets the lambda matrix for the given building. This means that we get a dictionary index by zone
    where we get the lambda for the given zone from start to end provided in the given interval. Taking the mean
    lambda for the interval.
    :param building: (string) building name 
    :param zones: (list str) The zones for which to get the data.
    :param start_utc: (datetime) the start of the data. timezone aware.
    :param end_utc: (datetime) the end of the data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :param datamanager_zones: None or {zone: Datamanager} If not provided it will create new ones but will add another
                                second of runtime.
    :return: {zone: pd.df columns="lambda" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    """
    return get_data_matrix(building, zones, start, end, interval, "prices", datamanager_zones)


def get_is_dr_matrix(building, zones, start, end, interval, datamanager_zones=None):
    """
    Gets the is_dr matrix for the given building. This means that we get a dictionary index by zone
    where we get the is_dr for the given zone from start to end provided in the given interval. Taking the mean
    round(is_dr) for the interval.
    :param building: (string) building name 
    :param zones: (list str) The zones for which to get the data.
    :param start_utc: (datetime) the start of the data. timezone aware.
    :param end_utc: (datetime) the end of the data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :param datamanager_zones: None or {zone: Datamanager} If not provided it will create new ones but will add another
                                second of runtime.
    :return: {zone: pd.Series with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    """
    temp_data = get_data_matrix(building, zones, start, end, interval, "is_dr", datamanager_zones)
    for iter_zone in zones:
        # TODO how to properly set values?
        temp_data[iter_zone].loc[:] = np.round(temp_data[iter_zone].values) == 1
    return temp_data


def get_data_matrix(building, zones, start, end, interval, data_type, datamanager_zones):
    """
    Gets the data matrix for the given building for the given data function.
    This means that we get a dictionary index by zone where we get the data for the given zone from start to end 
    in the given interval. Taking the mean data values for the interval.
    NOTE: The timeseries will start at exactly param:start but may not end at the param:end. It will end at the k
     which makes start + k*interval closest to param:end and smaller than param:end. 
    :param building: (string) building name 
    :param zones: (list str) The zones for which to get the data.
    :param start_utc: (datetime) the start of the comfortband data. timezone aware.
    :param end_utc: (datetime) the end of the comfortband data. timezone aware. (INCLUSIVE)
    :param interval: (int) minute intervals in which to get the data
    :param data_type: (str) The type of data to get. For now we can choose 
                            ["occupancy", "comfortband", "safety", "prices", "lambda", "is_dr"]
    
    :param datamanager_zones: None or {zone: Datamanager} If not provided it will create new ones but will add another
                                second of runtime.
    :return: {zone: pd.df according to data_type with timeseries data from start to end (inclusive) with the interval 
    as frequency. will be in timezone of config file} The intervals will be found by taking the mean of data_type in the 
     interval. NOTE: the time series will have seconds=milliseconds=0. 
    """
    building_cfg = get_config(building)

    # make seconds and microseonds zero. So, we have round intervals und numbers and so that slicing will work without
    # excluding start times e.g. if timeseries has 00:45, 00:46 then this will allow us to start from 00:45 if we have
    # a start datetime of 00:45.5.
    # TODO maybe get rid of this ?
    # start = start.replace(second=0, microsecond=0)
    # end = end.replace(second=0, microsecond=0)

    # set all timezones for start and end
    start_utc = start.astimezone(tz=pytz.utc)
    end_utc = end.astimezone(tz=pytz.utc)
    start_cfg_timezone = start_utc.astimezone(tz=pytz.timezone(building_cfg["Pytz_Timezone"]))
    end_cfg_timezone = end_utc.astimezone(tz=pytz.timezone(building_cfg["Pytz_Timezone"]))


    all_zone_data = {}

    # gather and process data
    for iter_zone in zones:
        data_manager = datamanager_zones[iter_zone]
        zone_data = []
        zone_start_cfg_timezone = start_cfg_timezone
        # have to get data in day intervals.
        while zone_start_cfg_timezone.date() <= end_cfg_timezone.date():
            # Deciding the type of data to get.
            if data_type == "occupancy":
                temp_data = data_manager.get_better_occupancy_config(zone_start_cfg_timezone)
            elif data_type == "comfortband":
                temp_data = data_manager.get_better_comfortband(zone_start_cfg_timezone)
            elif data_type == "safety":
                temp_data = data_manager.get_better_safety(zone_start_cfg_timezone)
            elif data_type == "prices":
                temp_data = data_manager.get_better_prices(zone_start_cfg_timezone)
            elif data_type == "lambda":
                temp_data = data_manager.get_better_lambda(zone_start_cfg_timezone)
            elif data_type == "is_dr":
                temp_data = data_manager.get_better_is_dr(zone_start_cfg_timezone)
            else:
                raise Exception("Bad data func argument passed: %s" % data_type)
            zone_data.append(temp_data)
            zone_start_cfg_timezone += datetime.timedelta(days=1)
        # Process data
        # Following makes sure we don't have the last datapoint as duplicate.
        # Since the methods get from start + one day.
        reduced_zone_data = reduce(lambda x, y: pd.concat([x.iloc[:-1], y]), zone_data)

        # Get the offset by which to shift the resampling to make the resampling start at the actual start time.
        # https://stackoverflow.com/questions/33446776/how-to-resample-starting-from-the-first-element-in-pandas
        # https://stackoverflow.com/questions/33575758/resampling-in-pandas?rq=1
        offset_microsecond = (start_cfg_timezone - start_cfg_timezone.replace(minute=0, second=0,
                                                                              microsecond=0)).total_seconds() % (
                             interval * 60)
        offset_minute = offset_microsecond / (60.)
        reduced_zone_data = reduced_zone_data.resample(str(interval) + "T", base=offset_minute).mean()

        # TODO SOOOOO WEIRD
        # TODO SOOOOO WEIRD
        # TODO should be pandas method, and there has to be a way to only do this with offsets.
        # NEED TO ADD MINUTES TO ACCOUNT FOR MICROSECOND MESS UP.
        reduced_zone_data = reduced_zone_data.loc[
                            start_cfg_timezone - datetime.timedelta(minutes=1):end_cfg_timezone + datetime.timedelta(
                                minutes=1)]
        # setting index which has correct microseconds. Issue is that pandas uses the offset given and adds
        # a seemingly arbitrary microsecond precision. Hence, the start will not be in the index because it is off
        # by a fraction of a millisecond. This will ensure that the timeseries starts and ends exactly at start and end.
        reduced_zone_data.index = pd.date_range(start_cfg_timezone, end_cfg_timezone, freq=str(interval)+"T")

        # set the data
        all_zone_data[iter_zone] = reduced_zone_data

    return all_zone_data


# ============ THERMOSTAT FUNCTIONS ============


def has_setpoint_changed(tstat, setpoint_data, zone):
    """
    Checks if thermostats was manually changed and prints warning.
    :param tstat: Tstat object we want to look at.
    :param setpoint_data: dict which has keys {"heating_setpoint": bool, "cooling_setpoint": bool} and corresponds to
            the setpoint written to the thermostat by MPC.
    :param zone: Name of the zone to print correct messages.
    :return: Bool. Whether tstat setpoints are equal to setpoints written to tstat.
    """
    WARNING_MSG = "WARNING. %s has been manually changed in zone %s. Setpoint is at %s from expected %s. " \
                  "Setting override to False and intiatiating program stop."
    flag_changed = False
    if tstat.cooling_setpoint != setpoint_data["cooling_setpoint"]:
        flag_changed = True
        print(WARNING_MSG % ("cooling setpoint", zone, tstat.cooling_setpoint, setpoint_data["cooling_setpoint"]))
    if tstat.heating_setpoint != setpoint_data["heating_setpoint"]:
        flag_changed = True
        print(WARNING_MSG % ("heating setpoint", zone, tstat.heating_setpoint, setpoint_data["heating_setpoint"]))

    return flag_changed


def set_override_false(tstat):
    tstat.write({"override": False})


def get_thermostats(client, hod, building):
    """Gets the thermostats for given building.
    :param client: xbos client object
    :param hod: hod client object
    :param building: (string) building name
    :return {zone: tstat object}"""

    query = """SELECT ?uri ?zone FROM %s WHERE {
        ?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
        ?tstat bf:uri ?uri .
        ?tstat bf:controls/bf:feeds ?zone .
        };"""

    # Start of FIX for missing Brick query
    query = """SELECT ?zone ?uri FROM  %s WHERE {
              ?tstat rdf:type brick:Thermostat .
              ?tstat bf:controls ?RTU .
              ?RTU rdf:type brick:RTU .
              ?RTU bf:feeds ?zone. 
              ?zone rdf:type brick:HVAC_Zone .
              ?tstat bf:uri ?uri.
              };"""
    # End of FIX - delete when Brick is fixed
    building_query = query % building

    tstat_query_data = hod.do_query(building_query)['Rows']
    tstats = {tstat["?zone"]: Thermostat(client, tstat["?uri"]) for tstat in tstat_query_data}
    return tstats


def action_logic(cfg_zone, temperature_zone, action_zone):
    """
    Returns a message to send to a thermostat given the action. 
    :param cfg_zone: config file for zone.
    :param temperature_zone: (float) zone temperature
    :param action_zone: (int) action we want to choose as given by utils constants.
    :param safety_constraint_zone: (float dict) keys: "t_low", "t_high" and tells us current safety constraints.
    :return: {heating_setpoint: float, cooling_setpoint_float}. None if didn't succeed or bad action given.
    """
    # action "0" is Do Nothing, action "1" is Heating, action "2" is Cooling
    if action_zone == NO_ACTION:
        heating_setpoint = temperature_zone - cfg_zone["Advise"]["Minimum_Comfortband_Height"] / 2.
        cooling_setpoint = temperature_zone + cfg_zone["Advise"]["Minimum_Comfortband_Height"] / 2.

    # heating
    elif action_zone == HEATING_ACTION:
        heating_setpoint = temperature_zone + 2 * cfg_zone["Advise"]["Hysterisis"]
        cooling_setpoint = heating_setpoint + cfg_zone["Advise"]["Minimum_Comfortband_Height"]

    # cooling
    elif action_zone == COOLING_ACTION:
        cooling_setpoint = temperature_zone - 2 * cfg_zone["Advise"]["Hysterisis"]
        heating_setpoint = cooling_setpoint - cfg_zone["Advise"]["Minimum_Comfortband_Height"]

    else:
        return None

    msg = {"heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint}

    return msg


def safety_check(cfg_zone, temperature_zone, safety_constraint_zone, cooling_setpoint, heating_setpoint):
    """
    Returns a message to send to a thermostat which is checked against the safety constraints and correctly adjusted.
     Also, ensures that the minimium_comfortband_height is ensured if violated. 
    :param cfg_zone: the zone config file
    :param temperature_zone: (float) the current temperature of the zone.
    :param safety_constraint_zone: {"t_high": (float) safety cooling setpoint, "t_low": (float) safety heating setpoint}
    :param cooling_setpoint: (float) the cooling setpoint to be tested
    :param heating_setpoint: (float) the heating setpoint to be tested
    :return: {heating_setpoint: float, cooling_setpoint_float}. None if safety constraints and comfortband could not be
                set correctly while keeping the action the setpoints were set to take.
    """


    # check if setpoints are set to cool, heat, or do nothing
    is_set_cooling = cooling_setpoint < temperature_zone - cfg_zone["Advise"]["Hysterisis"]
    is_set_heating = heating_setpoint > temperature_zone + cfg_zone["Advise"]["Hysterisis"]
    is_set_no_action = not (is_set_heating or is_set_cooling)

    cooling_setpoint_safety = safety_constraint_zone['t_high']
    heating_setpoint_safety = safety_constraint_zone['t_low']

    assert heating_setpoint < cooling_setpoint
    assert heating_setpoint_safety < cooling_setpoint_safety

    # Ensure proper rounding and also Check if comfortband is violated and act if it is.
    # Make sure that if the setpoint band is either lower or above the current temperature, that it stays there.
    comfortband_height_violated = cooling_setpoint - heating_setpoint < cfg_zone["Advise"]["Minimum_Comfortband_Height"]

    # want to cool.
    if is_set_cooling:
        if comfortband_height_violated:
            heating_setpoint = cooling_setpoint - cfg_zone["Advise"]["Minimum_Comfortband_Height"]

        # round to integers since the thermostats round internally and could up and make us not cool
        heating_setpoint = math.floor(heating_setpoint)
        cooling_setpoint = math.floor(cooling_setpoint)

    # want to heat
    elif is_set_heating:
        if comfortband_height_violated:
            cooling_setpoint = heating_setpoint + cfg_zone["Advise"]["Minimum_Comfortband_Height"]

        # round to integers since the thermostats round internally and could up and make us not heat.
        heating_setpoint = math.ceil(heating_setpoint)
        cooling_setpoint = math.ceil(cooling_setpoint)

    # Do nothing.
    else:
        if comfortband_height_violated:
            diff = cooling_setpoint - heating_setpoint
            cooling_setpoint = cooling_setpoint - diff / 2. + cfg_zone["Advise"]["Minimum_Comfortband_Height"] / 2.
            heating_setpoint = heating_setpoint + diff / 2. - cfg_zone["Advise"]["Minimum_Comfortband_Height"] / 2.

        # round to integers since the thermostats round internally.
        heating_setpoint = math.floor(heating_setpoint)
        cooling_setpoint = math.ceil(cooling_setpoint)

    # making sure that we are not exceeding the Safety temps and correctly adjust.
    if heating_setpoint < heating_setpoint_safety:
        diff = heating_setpoint_safety - heating_setpoint
        cooling_setpoint += diff
        heating_setpoint += diff

    if cooling_setpoint > cooling_setpoint_safety:
        diff = cooling_setpoint - cooling_setpoint_safety
        cooling_setpoint -= diff
        heating_setpoint -= diff

    # Check if adjusting setpoint temperatures according to safety constraints made the setpoints loose the action it
    # was set to.
    # If was cooling but not cooling anymore.
    if is_set_cooling and cooling_setpoint >= temperature_zone - cfg_zone["Advise"]["Hysterisis"]:
        return None
    # If was set heating but not anymore.
    if is_set_heating and heating_setpoint <= temperature_zone + cfg_zone["Advise"]["Hysterisis"]:
        return None
    # If it was no action but will now either heat or cool.
    if is_set_no_action and (heating_setpoint > temperature_zone + cfg_zone["Advise"]["Hysterisis"] or
                             cooling_setpoint < temperature_zone - cfg_zone["Advise"]["Hysterisis"]):
        return None

    return {"cooling_setpoint": cooling_setpoint, "heating_setpoint": heating_setpoint}


# ========== ACUTATION HELPER FUNCTIONS ===========


# TODO fix this function up.
def actuate_lights(now_cfg_timezone, cfg_building, cfg_zone, zone, client):
    """
    
    :param now_cfg_timezone: 
    :param cfg_building: 
    :param cfg_zone: 
    :param zone: 
    :param client: 
    :return: 
    """
    if is_DR(now_cfg_timezone, cfg_building):
        print("NOTE: Running the lights script from zone %s." % zone)
        lights.lights(building=cfg_building["Building"], client=client, actuate=True)
        # Overriding the lights.
        cfg_zone["Actuate_Lights"] = False
        with open("Buildings/" + cfg_building["Building"] + "/ZoneConfigs/" + zone + ".yml", 'wb') as ymlfile:
            yaml.dump(cfg_zone, ymlfile)


# ============ PLOTTING FUNCTIONS ============


def plotly_figure(G, path=None):
    pos = graphviz_layout(G, prog='dot')

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    my_annotations = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        my_annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                xref='x',
                yref='y',
                text="" + G.get_edge_data(edge[0], edge[1])['action'] +
                     G.get_edge_data(edge[0], edge[1])['model_type'][0][0],
                # TODO for multigraph use [0] to get the frist edge. Also, only using the first letter to identify the model.
                showarrow=False,
                arrowhead=2,
                ax=0,
                ay=0
            )
        )

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.Marker(
            showscale=False,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'].append(x)
        node_trace['y'].append(y)

        node_info = "Time: +{0}<br>Temps: {1}<br>Usage Cost: {2}".format(node.time,
                                                                         node.temps,
                                                                         G.node[node]['usage_cost'])

        node_trace['text'].append(node_info)

        if path is None:
            node_trace['marker']['color'].append(G.node[node]['usage_cost'])
        elif node in path:
            node_trace['marker']['color'].append('rgba(255, 0, 0, 1)')
        else:
            node_trace['marker']['color'].append('rgba(0, 0, 255, 1)')

    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont=dict(size=16),
                        showlegend=False,
                        width=650,
                        height=650,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=my_annotations,
                        xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    return fig


# ========= Multithreading ========

class Barrier:
    """Class which behaves like python3's Barrier class.
    NOTE: Never change any of the internal logic or set variables after they were set in the init."""

    def __init__(self, num_threads):
        import threading
        self.num_threads = num_threads
        self.count = 0
        self.mutex = threading.Semaphore(1)
        self.barrier = threading.Semaphore(0)

        self.is_set = True

    def wait(self):
        """Behaves like wait function from Barrier class. Make all threads wait together and then release them."""

        self.mutex.acquire()
        if not self.is_set:
            self.reset()
        self.mutex.release()

        # increment counter by one to indicate that another thread is waiting now.
        self.mutex.acquire()
        self.count = self.count + 1
        self.mutex.release()

        # check if enough threads are waiting. If enough are waiting, the barrier will be opened
        if self.count == self.num_threads:
            self.barrier.release()

        # if not enough threads are waiting, make the thread wait for the barrier to be released in the if statement.
        self.barrier.acquire()
        # release the barrier so other threads can use it
        self.barrier.release()

        # we set the flag to false. However, this should be fine since every thread should already be past
        # the if statement that checks whether the Barrier is_set.
        self.mutex.acquire()
        if self.is_set:
            self.is_set = False
        self.mutex.release()

    def reset(self):
        """Resets the barrier class."""
        self.count = 0
        self.barrier.acquire()
        self.is_set = True


if __name__ == "__main__":
    # bldg = "csu-dominguez-hills"
    # inside, outside = get_raw_data(building=bldg, days_back=20, force_reload=True)
    # use_data = {}
    # for zone, zone_data in inside.items():
    #     if zone != "HVAC_Zone_Please_Delete_Me":
    #         use_data[zone] = zone_data
    #         print(zone)
    #         print(zone_data[zone_data["action"] == 2].shape)
    #         print(zone_data[zone_data["action"] == 5].shape)
    #
    # t_man = ThermalDataManager.ThermalDataManager({"Building": bldg}, client=get_client())
    # outside = t_man._preprocess_outside_data(outside.values())
    # print("inside")
    # th_data = t_man._preprocess_thermal_data(use_data, outside, True)

    #
    # import pickle
    # with open("u_p", "r") as f:
    #     th = pickle.load(f)
    #
    # zone = "HVAC_Zone_SAC_2101"
    # zone_data = th[zone]
    # print(zone_data[zone_data["action"] == 5].shape)
    # print(zone_data[zone_data["action"] == 2].shape)

    # ===== TESTING THE BARRIER
    # test_barrier = True
    # if test_barrier:
    #     barrier = Barrier(2)
    #     import time
    #     import threading
    #
    #
    #     def func1():
    #         time.sleep(3)
    #         #
    #         barrier.wait()
    #         #
    #         print('Working from func1')
    #         return
    #
    #
    #     def func2():
    #         time.sleep(5)
    #         #
    #         barrier.wait()
    #         #
    #         print('Working from func2')
    #         return
    #
    #
    #     threading.Thread(target=func1).start()
    #     threading.Thread(target=func2).start()
    #
    #     time.sleep(6)
    #
    #     # check if reset
    #     threading.Thread(target=func1).start()
    #     threading.Thread(target=func2).start()

    # a_dict = {"a": True, "b": False}
    # b_dict = {"b": True, 'a': True}
    # print(and_dictionary(a_dict, b_dict))


    BUILDING = "ciee"
    ZONES = get_zones(BUILDING)
    ZONE = ZONES[0]
    print(ZONE)
    building_config = get_config(BUILDING)
    zone_config = get_zone_config(BUILDING, ZONE)
    zone_config = get_zone_config(BUILDING, ZONE)

    heating_setpoint = 74
    cooling_setpoint = 75

    safety = {"t_low": 60, "t_high": 79}

    curr_temp = 73

    start_simulation = datetime.datetime(year=2018, month=8, day=23, hour=10, minute=0)
    end_simulation = start_simulation + datetime.timedelta(hours=4)

    cfg_building = get_config(BUILDING)
    cfg_timezone = pytz.timezone(cfg_building["Pytz_Timezone"])
    start_simulation = cfg_timezone.localize(start_simulation)
    end_simulation = cfg_timezone.localize(end_simulation)



    data_manager = DataManager.DataManager(building_config, zone_config, get_client(), ZONE,
                 now=None)

    thermal_data_manager = ThermalDataManager.ThermalDataManager(building_config, get_client(), interval=5)

    print(get_outside_temperatures(building_config, start_simulation, end_simulation, data_manager, thermal_data_manager, 15))
    # print(safety_check(zone_config, curr_temp, safety, cooling_setpoint, heating_setpoint))

    # start = get_utc_now() + datetime.timedelta(hours=1)
    # end = start + datetime.timedelta(hours=7)

    # # client = choose_client()
    # client = None

    # data_manager = DataManager.DataManager(building_config, zone_config, client, ZONE,
    #              now=None)
    #
    # thermal_data_manager = ThermalDataManager.ThermalDataManager(building_config, client, interval=5)
    #
    # print(get_outside_temperatures(building_config, start, end, data_manager, thermal_data_manager))
    #
    # datamanager_zones = get_zone_data_managers(BUILDING, ZONES, start, client)
    #
    # import time
    #
    # s_time = time.time()
    # print(get_is_dr_matrix(BUILDING, ZONES, start, end, 15, datamanager_zones=datamanager_zones))
    # print(time.time() - s_time)
