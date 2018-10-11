import datetime
from datetime import timedelta

import numpy as np
import sys
sys.path.append("./..")
import utils
import pandas as pd
import pytz
import yaml
from xbos import get_client
from xbos.services import mdal
from xbos.services.hod import HodClient


# TODO add energy data acquisition
# TODO FIX DAYLIGHT TIME CHANGE PROBLEMS



def get_inside_data(self, start, end):
    """Get thermostat status and temperature and outside temperature for thermal model.
    :param start: (datetime) time to start. in UTC time.
    :param end: (datetime) time to end. in UTC time.
    :return outside temperature has freq of 15 min and
                pd.df columns["tin", "action"] has freq of self.window_size. """

    # following queries are for the whole building.
    thermostat_status_query = """SELECT ?zone ?uuid FROM %s WHERE { 
          ?tstat rdf:type brick:Thermostat .
          ?tstat bf:hasLocation/bf:isPartOf ?location_zone .
          ?location_zone rdf:type brick:HVAC_Zone .
          ?tstat bf:controls ?RTU .
          ?RTU rdf:type brick:RTU . 
          ?RTU bf:feeds ?zone. 
          ?zone rdf:type brick:HVAC_Zone . 
          ?tstat bf:hasPoint ?status_point .
          ?status_point rdf:type brick:Thermostat_Status .
          ?status_point bf:uuid ?uuid.
        };"""

    # Start of FIX for missing Brick query
    thermostat_status_query = """SELECT ?zone ?uuid FROM  %s WHERE {
                             ?tstat rdf:type brick:Thermostat .
                             ?tstat bf:controls ?RTU .
                             ?RTU rdf:type brick:RTU .
                             ?RTU bf:feeds ?zone. 
                             ?zone rdf:type brick:HVAC_Zone .
                             ?tstat bf:hasPoint ?status_point .
                              ?status_point rdf:type brick:Thermostat_Status .
                              ?status_point bf:uuid ?uuid.
                             };"""
    # End of FIX - delete when Brick is fixed

    thermostat_temperature_query = """SELECT ?zone ?uuid FROM %s WHERE { 
          ?tstat rdf:type brick:Thermostat .
          ?tstat bf:hasLocation/bf:isPartOf ?location_zone .
          ?location_zone rdf:type brick:HVAC_Zone .
          ?tstat bf:controls ?RTU .
          ?RTU rdf:type brick:RTU . 
          ?RTU bf:feeds ?zone. 
          ?zone rdf:type brick:HVAC_Zone . 
          ?tstat bf:hasPoint ?thermostat_point .
          ?thermostat_point rdf:type brick:Temperature_Sensor .
          ?thermostat_point bf:uuid ?uuid.
        };"""

    # Start of FIX for missing Brick query
    thermostat_temperature_query = """SELECT ?zone ?uuid FROM  %s WHERE {
                      ?tstat rdf:type brick:Thermostat .
                      ?tstat bf:controls ?RTU .
                      ?RTU rdf:type brick:RTU .
                      ?RTU bf:feeds ?zone. 
                      ?zone rdf:type brick:HVAC_Zone .
                      ?tstat bf:hasPoint ?thermostat_point  .
                      ?thermostat_point rdf:type brick:Temperature_Sensor .
                      ?thermostat_point bf:uuid ?uuid.
                      };"""
    # End of FIX - delete when Brick is fixed

    # get query data
    temp_thermostat_query_data = {
        "tstat_temperature": self.hod_client.do_query(thermostat_temperature_query % self.building)["Rows"],
        "tstat_action": self.hod_client.do_query(thermostat_status_query % self.building)["Rows"],
    }

    # give the thermostat query data better structure for later loop. Can index by zone and then get uuids for each
    # thermostat attribute.
    thermostat_query_data = {}
    for tstat_attr, attr_dicts in temp_thermostat_query_data.items():
        for dict in attr_dicts:
            if dict["?zone"] not in thermostat_query_data:
                thermostat_query_data[dict["?zone"]] = {}
            thermostat_query_data[dict["?zone"]][tstat_attr] = dict["?uuid"]

    # get the data for the thermostats for each zone.
    mdal_client = mdal.MDALClient("xbos/mdal", client=self.client)
    zone_thermal_data = {}
    for zone, dict in thermostat_query_data.items():
        mdal_query = {'Composition': [dict["tstat_temperature"], dict["tstat_action"]],
                      'Selectors': [mdal.MEAN, mdal.MEAN]
            , 'Time': {'T0': start.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                       'T1': end.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                       'WindowSize': str(self.window_size) + 'min',
                       'Aligned': True}}

        # get the thermostat data
        df = utils.get_mdal_data(mdal_client, mdal_query)
        zone_thermal_data[zone] = df.rename(
            columns={dict["tstat_temperature"]: 't_in', dict["tstat_action"]: 'action'})

    return zone_thermal_data

def preprocess_zone_thermal_data(self, thermal_model_data, evaluate_preprocess=False):
    """
    Preprocess data for one zone.
    :param thermal_model_data: pd.df columns: t_in','t_out', 'action', [other zone temperatures]
    :param evaluate_preprocess: If flag on, adds more fields to understand thermal data better.
    :return: pd.df columns: t_in', 't_next', 'dt','t_out', 'action', 'a1', 'a2', [other mean zone temperatures]
    """
    # Finding all points where action changed.
    thermal_model_data['change_of_action'] = (thermal_model_data['action'].diff(1) != 0).astype(
        'int').cumsum()

    # following adds the fields "time", "dt" etc such that we accumulate all values where we have consecutively the same action.

    # NOTE: Only dropping nan value during gathering stage and after that. Before then not doing it because
    # we want to keep the times contigious.
    data_list = []
    for j in thermal_model_data.change_of_action.unique():
        for i in range(0, thermal_model_data[thermal_model_data['change_of_action'] == j].shape[0],
                       self.interval):

            # get dfs to process.
            dfs = thermal_model_data[thermal_model_data['change_of_action'] == j][i:i + self.interval + 1]

            # we only look at intervals where the last and first value for T_in are not Nan.
            dfs.dropna(subset=["t_in"])
            zone_col_filter = ["zone_temperature_" in col for col in dfs.columns]
            temp_data_dict = {'time': dfs.index[0],
                              't_in': dfs['t_in'][0],
                              't_next': dfs['t_in'][-1],
                              'dt': (dfs.index[-1] - dfs.index[0]).seconds / 60,
                              't_out': dfs['t_out'].mean(),  # mean does not count Nan values
                              'action': dfs['action'][
                                  0]}

            # add fields to debug thermal data.
            if evaluate_preprocess:
                temp_data_dict["t_max"] = max(dfs["t_in"])
                temp_data_dict["t_min"] = min(dfs["t_in"])

            for temperature_zone in dfs.columns[zone_col_filter]:
                # mean does not count Nan values
                temp_data_dict[temperature_zone] = dfs[temperature_zone].mean()
            data_list.append(temp_data_dict)

    thermal_model_data = pd.DataFrame(data_list).set_index('time')

    thermal_model_data = thermal_model_data.dropna()  # final drop. Mostly if the whole interval for the zones or t_out were nan.

    return thermal_model_data

def preprocess_thermal_data(self, zone_data, outside_data, evaluate_preprocess=False):
    """Preprocesses the data for the thermal model.
    :param zone_data: dict{zone: pd.df columns["tin", "action"]}
    :param outside_data: pd.df columns["tout"].
    NOTE: outside_data freq has to be a multiple of zone_data frequency and has to have a higher freq.

    :returns {zone: pd.df columns: t_in', 't_next', 'dt','t_out', 'action', [other mean zone temperatures]}
             where t_out and zone temperatures are the mean values over the intervals. """

    # thermal data preprocess starts here
    all_temperatures = pd.concat([tstat_df["t_in"] for tstat_df in zone_data.values()], axis=1)
    all_temperatures.columns = ["zone_temperature_" + zone for zone in zone_data.keys()]
    zone_thermal_model_data = {}

    for zone in zone_data.keys():
        # Putting together outside and zone data.
        actions = zone_data[zone]["action"]
        thermal_model_data = pd.concat([all_temperatures, actions, outside_data],
                                       axis=1)  # should be copied data according to documentation
        thermal_model_data = thermal_model_data.rename(columns={"zone_temperature_" + zone: "t_in"})

        # Interpolate t_out values linearly, since t_out does not need to be as accurate as inside data.
        thermal_model_data["t_out"] = thermal_model_data["t_out"].interpolate()

        # Preprocessing data. Concatinating all time contigious datapoints which have the same action.
        zone_thermal_model_data[zone] = self.preprocess_zone_thermal_data(thermal_model_data,
                                                                          evaluate_preprocess=evaluate_preprocess)

        print('one zone preproccessed')
    return zone_thermal_model_data

def thermal_data(self, start=None, end=None, days_back=60, evaluate_preprocess=False):
    """
    :param start: In UTC time.
    :param end: In UTC time.
    :param days_back: if start is None, then we set start to end - timedelta(days=days_back).
    :param evaluate_preprocess: Whether to add the fields t_max, t_min, t_mean, t_std to the preprocessed data.
    :return: pd.df {zone: pd.df columns: t_in', 't_next', 'dt','t_out', 'action', 'a1', 'a2', [other mean zone temperatures]}
             where t_out and zone temperatures are the mean values over the intervals.
             a1 is whether heating and a2 whether cooling.
    """
    if end is None:
        end = utils.get_utc_now()
    if start is None:
        start = end - timedelta(days=days_back)
    z, o = self.get_inside_data(start, end), self.get_outside_data(start, end)
    # Take mean of all weather stations.
    o = self.preprocess_outside_data(o.values())
    print("Received Thermal Data from MDAL.")
    return self.preprocess_thermal_data(z, o, evaluate_preprocess)




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


if __name__ == '__main__':
    building = "ciee"
    end = datetime.datetime(2018, 10, 5, 0, 37, 27, 345097, tzinfo=pytz.utc)
    start = datetime.datetime(2018, 10, 3, 13, 37, 27, 345097, tzinfo=pytz.utc)

    print("start", start)
    print("end", end)
    print(get_outside_data(building, start, end, 15))