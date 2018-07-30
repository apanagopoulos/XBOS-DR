# util methods for the linear program

from xbos import get_client

import os
import yaml
import datetime
import pytz

from DataManager import DataManager


# UTILITY CONSTANTS

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
                    current time.
    :return: {zone: DataManager}
    """
    building_cfg = get_config(building)
    zones = get_zones(building)
    client = choose_client(building_cfg)
    if now is None:
        now = get_utc_now()
    zone_managers = {}
    for zone in zones:
        zone_config = get_zone_config(building, zone)
        zone_managers[zone] = DataManager(controller_cfg=building_cfg, advise_cfg=zone_config, client=client,
                                          zone=zone, now=now)
    return zone_managers



def get_data_matrix(zone_data_managers):
    pass

# ======= END DATA METHODS ======

if __name__ == "__main__":
    BUILDING = "ciee"
    ZONES = get_zones(BUILDING)
    print(ZONES)
    print(get_zone_data_managers(BUILDING))