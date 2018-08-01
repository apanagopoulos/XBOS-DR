# util methods for the linear program

from xbos import get_client

import os
import yaml
import datetime
import pytz
import pandas as pd

from DataManager import DataManager


# UTILITY CONSTANTS

NO_ACTION = 0
HEATING_ACTION = 1
COOLING_ACTION = 2
FAN = 3
TWO_STAGE_HEATING_ACTION = 4
TWO_STAGE_COOLING_ACTION = 5

SERVER_DIR_PATH = UTILS_FILE_PATH = os.path.dirname(__file__)  # this is true for now

# ======= DATE FUNCTIONS ======

def get_utc_now():
    """Gets current time in utc time.
    :return Datetime in utctime zone"""
    return datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))

# =============

# ======= DATA METHODS ======

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
        print("ERROR: No config file for building %s with path %s" % (building, config_path))
        return
    return cfg


def get_zone_config(building, zone):
    config_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + "ZoneConfigs/" + zone + ".yml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.load(f)
    except:
        print("ERROR: No config file for building %s and zone % s with path %s" % (building, zone, config_path))
        return
    return cfg

def choose_client(cfg=None):
    if cfg is not None and cfg["Server"]:
        client = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        client = get_client()
    return client

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



# ======= END DATA METHODS ======

if __name__ == "__main__":
    s = get_utc_now() + datetime.timedelta(hours=7)
    e = s + datetime.timedelta(hours=1)
    print(s)
    print(e)

    BUILDING = "ciee"
    ZONES = get_zones(BUILDING)
    # data_types = ["prices"] #["occupancy", "comfortband", "safety", "prices"]
    data_types = ["occupancy", "comfortband", "safety", "prices"]
    for data_type in data_types:
        print(data_type)
        print(get_data_matrix(building=BUILDING, start=s, end=e, interval=15, data_func=data_type))