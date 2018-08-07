# this is the plotter for the MPC graph

import datetime
import os
import string

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytz
import yaml
from xbos import get_client
from xbos.devices.thermostat import Thermostat

# be careful of circular import.
# https://stackoverflow.com/questions/11698530/two-python-modules-require-each-others-contents-can-that-work
from ThermalDataManager import ThermalDataManager

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
    return date.replace(hour=datetime_time.hour, minute=datetime_time.minute, second=0, microseconds=0)
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


# ============ DATA FUNCTIONS ============

def round_increment(data, precision=0.05):
    """Round to nearest increment of precision.
    :param data: np.array of floats or single float
    :param precision: (float) the increment to round to
    :return (np.array or float) of rounded floats."""
    # source for rounding: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    return precision * np.round(data / float(precision))


def is_cooling(action_data):
    """Returns boolen area of actions which were cooling (either two or single stage).
    :param action_data: np.array or pd.series"""
    return (action_data == COOLING_ACTION) | (action_data == TWO_STAGE_COOLING_ACTION)


def is_heating(action_data):
    """Returns boolen area of actions which were heating (either two or single stage).
    :param action_data: np.array or pd.series"""
    return (action_data == HEATING_ACTION) | (action_data == TWO_STAGE_HEATING_ACTION)


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
        raise Exception("ERROR: No config file for building %s and zone % s with path %s" % (building, zone, config_path))
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
        dataManager = ThermalDataManager(cfg, client)
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
        dataManager = ThermalDataManager(cfg, client)

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

# TODO Finsih this up once i have more energy. Make it return a dataframe.
def get_outside_temperatures(building_config, start, end, data_manager, thermal_data_manager):
    """
    Get outside weather from start to end. Will combine historic and weather predictions data when necessary.
    :param start: datetime timezone aware
    :param end: datetime timezone aware
    :return: {int hour: float temperature}
    """
    # we might have that the given now is before the actual current time
    # hence need to get historic data and combine with weather predictions.

    # For finding out if start or/and end are before or after the current time.
    utc_now = get_utc_now()

    # Set start and end to correct timezones
    cfg_timezone = pytz.timezone(building_config["Pytz_Timezone"])
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

    # Populating the outside_temperatures dictionary for MPC use. Ouput is in cfg timezone.
    outside_temperatures = {}
    if future_start_utc is not None:
        # TODO implement end for weather_fetch
        future_weather = data_manager.weather_fetch(start=future_start_utc)
        outside_temperatures = future_weather

    # Combining historic data with outside_temperatures correctly if exists.
    if historic_start_utc is not None:
        historic_weather = thermal_data_manager._get_outside_data(historic_start_utc,
                                                                  historic_start_utc, inclusive=True)
        historic_weather = thermal_data_manager._preprocess_outside_data(historic_weather.values())

        # Down sample the historic weather to hourly entries, and take the mean for each hour.
        historic_weather = historic_weather.groupby([pd.Grouper(freq="1H")])["t_out"].mean()

        # Convert historic_weather to cfg timezone.
        historic_weather.index = historic_weather.index.tz_convert(tz=building_config["Pytz_Timezone"])

        # Popluate the outside_temperature array. If we have the simulation time in the past and future then
        # we will take a weighted averege of the historic and future temperatures in the hour in which
        # historic_end and future_start happen.
        for row in historic_weather.iteritems():
            row_time, t_out = row[0], row[1]

            # taking a weighted average of the past and future outside temperature since for now
            # we only have one outside temperature per hour.
            if row_time.hour in outside_temperatures and \
                            row_time.hour == historic_end_cfg_timezone.hour:

                future_t_out = outside_temperatures[row_time.hour]

                # Checking if start and end are in the same hour, because then we have to weigh the temperature by
                # less.
                if historic_end_cfg_timezone.hour ==\
                       historic_start_cfg_timezone.hour:
                    historic_weight = (historic_end_cfg_timezone - historic_start_cfg_timezone).seconds // 60
                else:
                    historic_weight = historic_end_cfg_timezone.minute
                if future_start_cfg_timezone.hour ==\
                        future_end_cfg_timezone.hour:
                    future_weight = (future_end_cfg_timezone - future_start_cfg_timezone).seconds // 60
                else:
                    # the remainder of the hour.
                    future_weight = 60 - future_start_cfg_timezone.minute
                # Normalize
                total_weight = future_weight + historic_weight
                future_weight /= float(total_weight)
                historic_weight /= float(total_weight)

                outside_temperatures[row_time.hour] = future_weight * future_t_out + \
                                                      historic_weight * float(t_out)

            else:
                outside_temperatures[row_time.hour] = float(t_out)

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

def get_zone_data_managers(building, now=None):
    """
    Gets the data managers for each zone as a dictionary.
    :param building: (str) Building name 
    :param now: (datetime) the time for which the datamanager should return data. If None, we will use the 
                    current time. Timezone aware.
    :return: {zone: DataManager}
    """
    # make now utc for datamanager
    if now is None:
        now = get_utc_now()

    utc_now = now.astimezone(tz=pytz.utc)

    building_cfg = get_config(building)
    zones = get_zones(building)
    client = choose_client(building_cfg)

    zone_managers = {}
    for zone in zones:
        zone_config = get_zone_config(building, zone)
        zone_managers[zone] = DataManager(controller_cfg=building_cfg, advise_cfg=zone_config, client=client,
                                          zone=zone, now=utc_now)
    return zone_managers

def get_occupancy_matrix(building, start, end, interval):
    """
    Gets the occupancy matrix for the given building. This means that we get a dictionary index by zone
    where we get the occupancy for the given zone from start to end provided in the given interval. Taking the mean
    occupancy for the interval.
    :param building: (string) buidling name 
    :param start_utc: (datetime) the start of the occupancy data. timezone aware.
    :param end_utc: (datetime) the end of the occupancy data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :return: {zone: pd.df columns="occ" with timeseries data from start to end (inclusive) with the interval as 
     frequency. will be in timezone of config file} The intervals will be found by taking the mean occupancy in the 
     interval. 
    """
    # TODO implement archiver zone occupancy with predictions.

    return get_data_matrix(building, start, end, interval, "occupancy")

def get_comfortband_matrix(building, start, end, interval):
    """
    Gets the comfortband matrix for the given building. This means that we get a dictionary index by zone
    where we get the comfortband for the given zone from start to end provided in the given interval. Taking the mean
    comfortband for the interval.
    :param building: (string) buidling name 
    :param start_utc: (datetime) the start of the comfortband data. timezone aware.
    :param end_utc: (datetime) the end of the comfortband data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :return: {zone: pd.df columns="t_high", "t_low" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    The intervals will be found by taking the mean comfortband temperatures in the interval. 
    """
    return get_data_matrix(building, start, end, interval, "comfortband")

def get_safety_matrix(building, start, end, interval):
    """
    Gets the safety matrix for the given building. This means that we get a dictionary index by zone
    where we get the safety for the given zone from start to end provided in the given interval. Taking the mean
    safety for the interval.
    :param building: (string) buidling name 
    :param start_utc: (datetime) the start of the safety data. timezone aware.
    :param end_utc: (datetime) the end of the safety data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :return: {zone: pd.df columns="t_high", "t_low" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    The intervals will be found by taking the mean safety temperatures in the interval.  
    """
    return get_data_matrix(building, start, end, interval, "safety")

def get_price_matrix(building, start, end, interval):
    """
    Gets the price matrix for the given building. This means that we get a dictionary index by zone
    where we get the price for the given zone from start to end provided in the given interval. Taking the mean
    price for the interval.
    :param building: (string) building name 
    :param start_utc: (datetime) the start of the data. timezone aware.
    :param end_utc: (datetime) the end of the data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :return: {zone: pd.df columns="price" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    The intervals will be found by taking the mean price in the interval.  
    """
    return get_data_matrix(building, start, end, interval, "prices")

def get_lambda_matrix(building, start, end, interval):
    """
    Gets the lambda matrix for the given building. This means that we get a dictionary index by zone
    where we get the lambda for the given zone from start to end provided in the given interval. Taking the mean
    lambda for the interval.
    :param building: (string) building name 
    :param start_utc: (datetime) the start of the data. timezone aware.
    :param end_utc: (datetime) the end of the data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :return: {zone: pd.df columns="lambda" with timeseries data from start to end (inclusive) 
    with the interval as frequency. will be in timezone of config file} 
    The intervals will be found by taking the mean price in the interval.  
    """
    return get_data_matrix(building, start, end, interval, "prices")

def get_data_matrix(building, start, end, interval, data_func):
    """
    Gets the data matrix for the given building for the given data funciton.
    This means that we get a dictionary index by zone where we get the data for the given zone from start to end 
    in the given interval. Taking the mean data values for the interval.
    :param building: (string) building name 
    :param start_utc: (datetime) the start of the comfortband data. timezone aware.
    :param end_utc: (datetime) the end of the comfortband data. timezone aware.
    :param interval: (int) minute intervals in which to get the data
    :param data_func: (str) The type of data to get. For now we can choose 
                            ["occupancy", "comfortband", "safety", "prices", "lambda"]
    :return: {zone: pd.df columns="t_low" "t_high" with timeseries data from start to end (inclusive) with the interval 
    as frequency. will be in timezone of config file} The intervals will be found by taking the mean comforband in the 
     interval. NOTE: the time series will have seconds=milliseconds=0. 
    """
    building_cfg = get_config(building)

    # make seconds and microseonds zero. So, we have round intervals und numbers and so that slicing will work without
    # excluding start times e.g. if timeseries has 00:45, 00:46 then this will allow us to start from 00:45 if we have
    # a start datetime of 00:45.5.
    # TODO maybe get rid of this ?
    start = start.replace(second=0, microsecond=0)
    end = end.replace(second=0, microsecond=0)

    # set all timezones for start and end
    start_utc = start.astimezone(tz=pytz.utc)
    end_utc = end.astimezone(tz=pytz.utc)
    start_cfg_timezone = start_utc.astimezone(tz=pytz.timezone(building_cfg["Pytz_Timezone"]))
    end_cfg_timezone = end_utc.astimezone(tz=pytz.timezone(building_cfg["Pytz_Timezone"]))

    zone_data_managers = get_zone_data_managers(building, start_utc)
    all_zone_data = {}
    # gather and process data
    for zone, data_manager in zone_data_managers.items():
        zone_data = []
        zone_start_cfg_timezone = start_cfg_timezone
        # have to get data in day intervals.
        while zone_start_cfg_timezone.date() <= end_cfg_timezone.date():
            # Deciding the type of data to get.
            if data_func == "occupancy":
                temp_data = data_manager.get_better_occupancy_config(zone_start_cfg_timezone)
            elif data_func == "comfortband":
                temp_data = data_manager.get_better_comfortband(zone_start_cfg_timezone)
            elif data_func == "safety":
                temp_data = data_manager.get_better_safety(zone_start_cfg_timezone)
            elif data_func == "prices":
                temp_data = data_manager.get_better_prices(zone_start_cfg_timezone)
            elif data_func == "lambda":
                temp_data = data_manager.get_better_lambda(zone_start_cfg_timezone)
            else:
                raise Exception("Bad data func argument passed: %s" % data_func)
            zone_data.append(temp_data)
            zone_start_cfg_timezone += datetime.timedelta(days=1)
        # Process data
        # Following makes sure we don't have the last datapoint as duplicate.
        # Since better_comfortband gets from start + one day.
        reduced_zone_data = reduce(lambda x, y: pd.concat([x.iloc[:-1], y]), zone_data)
        reduced_zone_data = reduced_zone_data.loc[start_cfg_timezone:end_cfg_timezone]


        # Get the offset by which to shift the resampling to make the resampling start at the actual start time.
        # https://stackoverflow.com/questions/33446776/how-to-resample-starting-from-the-first-element-in-pandas
        # https://stackoverflow.com/questions/33575758/resampling-in-pandas?rq=1
        offset = reduced_zone_data.index[0].minute % interval
        reduced_zone_data = reduced_zone_data.resample(str(interval) + "T", base=offset).mean()
        all_zone_data[zone] = reduced_zone_data

    return all_zone_data

# ============ THERMOSTAT FUNCTIONS ============


def has_setpoint_changed(tstat, setpoint_data, zone, building):
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

    test_barrier = True
    if test_barrier:
        barrier = Barrier(2)
        import time
        import threading


        def func1():
            time.sleep(3)
            #
            barrier.wait()
            #
            print('Working from func1')
            return


        def func2():
            time.sleep(5)
            #
            barrier.wait()
            #
            print('Working from func2')
            return


        threading.Thread(target=func1).start()
        threading.Thread(target=func2).start()

        time.sleep(6)

        # check if reset
        threading.Thread(target=func1).start()
        threading.Thread(target=func2).start()
