import datetime
from datetime import timedelta

import numpy as np
import utils
import pandas as pd
import pytz
import yaml
from xbos import get_client
from xbos.services import mdal
from xbos.services.hod import HodClient


# TODO add energy data acquisition
# TODO FIX DAYLIGHT TIME CHANGE PROBLEMS


class ThermalDataManager:
    """
    # Class that handles all the data fetching and some of the preprocess for data that is relevant to controller
    and which does not have to be fetched every 15 min but only once. 
    
    Time is always in UTC
    """

    def __init__(self, building_cfg, client, interval=5):
        """
        
        :param building_cfg: A dictionary which should include data for keys "Building" and "Interval_Length"
        :param client: An xbos client.
        :param interval: the interval in which to split thermal data actions.
        """
        self.building_cfg = building_cfg
        self.building = building_cfg["Building"]
        self.interval = interval  # the interval in which to split thermal data actions.
        self.client = client

        self.window_size = 1  # minutes. TODO should come from config.
        self.hod_client = HodClient("xbos/hod", self.client)  # TODO Potentially could incorporate this into MDAL query.

    def preprocess_outside_data(self, outside_data):
        """
    
        :param outside_data: (list) of pd.
         with col ["t_out"] for each weather station. 
        :return: pd.df with col ["t_out"]
        """

        # since the archiver had a bug while storing weather.gov data, nan values were stored as 32. We will
        # replace any 32 values with Nan. Since we usually have more than one weather station, we can then take the
        # average across those to compensate. If no other weather stations, then issue a warning that
        # we might be getting rid of real data and we won't be able to recover through the mean.
        if len(outside_data) == 1:
            print("WARNING: Only one weather station for selected region. We need to replace 32 values with Nan due to "
                  "past inconsistencies, but not enough data to compensate for the lost data by taking mean.")

        for i in range(len(outside_data)):
            temp_data = outside_data[i]["t_out"].apply(
                lambda t: np.nan if t == 32 else t)  # TODO this only works for fahrenheit now.
            outside_data[i]["t_out"] = temp_data

        # Note: Assuming same index for all weather station data returned by mdal
        final_outside_data = pd.concat(outside_data, axis=1).mean(axis=1)
        final_outside_data = pd.DataFrame(final_outside_data, columns=["t_out"])

        # outside temperature may contain nan values.
        # Hence, we interpolate values linearly,
        # since outside temperature does not need to be as accurate as inside data.
        final_outside_data = final_outside_data.interpolate()

        return final_outside_data

    def get_outside_data(self, start=None, end=None, inclusive=False):
        """Get outside temperature for thermal model.
        :param start: (datetime) time to start. in UTC time.
        :param end: (datetime) time to end. in UTC time.
        :param inclusive: (bool) whether the end time should be inclusive. 
                        which means that we get the time for the end, plus 15 min.
        :return ({uuid: (pd.df) (col: "t_out) outside_data})  outside temperature has freq of 15 min and
        pd.df columns["tin", "action"] has freq of self.window_size. """

        # add an interval, to make the end inclusive, which means that we get the time for the end, plus 15 min.
        if inclusive:
            end += datetime.timedelta(minutes=15)

        outside_temperature_query = """SELECT ?weather_station ?uuid FROM %s WHERE {
                                    ?weather_station rdf:type brick:Weather_Temperature_Sensor.
                                    ?weather_station bf:uuid ?uuid.
                                    };"""

        # get outside temperature data
        # TODO for now taking all weather stations and preprocessing it. Should be determined based on metadata.
        outside_temperature_query_data = self.hod_client.do_query(outside_temperature_query % self.building)["Rows"]

        outside_temperature_data = {}
        for weather_station in outside_temperature_query_data:
            # Get data from MDAL
            mdal_client = mdal.MDALClient("xbos/mdal", client=self.client)
            mdal_query = {
                'Composition': [weather_station["?uuid"]],
                # uuid from Mr.Plotter. should use outside_temperature_query_data["?uuid"],
                'Selectors': [mdal.MEAN]
                , 'Time': {'T0': start.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                           'T1': end.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                           'WindowSize': str(15) + 'min',
                           # TODO document that we are getting 15 min intervals because then we get fewer nan values.
                           'Aligned': True}}

            mdal_outside_data = utils.get_mdal_data(mdal_client, mdal_query)

            mdal_outside_data.columns = ["t_out"]
            outside_temperature_data[weather_station["?uuid"]] = mdal_outside_data

        return outside_temperature_data

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


if __name__ == '__main__':
    cfg = utils.get_config("ciee")
    dataManager = ThermalDataManager(cfg, get_client())

    to_from = pd.date_range(utils.get_utc_now() - datetime.timedelta(hours=1), utils.get_utc_now(), freq="1T")
    thermal_data = pd.DataFrame(columns=["t_in", "t_out", "action"], index=to_from)

    thermal_data["t_in"] = np.linspace(70, 80, thermal_data.shape[0])
    thermal_data["t_out"] = np.linspace(60, 90, thermal_data.shape[0])

    thermal_data["action"] = np.zeros(thermal_data.shape[0])
    thermal_data["action"][0:20] = 1
    thermal_data["action"][42:49] = np.nan
    thermal_data["action"][18:24] = 2

    print(thermal_data)
    print(dataManager.preprocess_zone_thermal_data(thermal_data, evaluate_preprocess=True))

    # with open("./Buildings/avenal-animal-shelter/avenal-animal-shelter.yml", 'r') as ymlfile:
    #     cfg = yaml.load(ymlfile)
    #
    # if cfg["Server"]:
    #     c = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    # else:
    #     c = get_client()
    #
    # dm = ControllerDataManager(controller_cfg=cfg, client=c)
    # import pickle
    # # fetching data here
    # z = dm.thermal_data(days_back=30)
    #
    # with open("demo_anmial_shelter", "wb") as f:
    #     pickle.dump(z, f)
    # print(z)
    # import pickle
    #
    # with open("Buildings/avenal-recreation-center/avenal-recreation-center.yml") as f:
    #     cfg = yaml.load(f)

    # if cfg["Server"]:
    #     c = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    # else:
    # c = get_client()
    #
    # now = datetime.datetime(year=2018, month=06, day=01).replace(tzinfo=pytz.timezone("UTC"))
    #
    # dataManager = ThermalDataManager(cfg, c)
    #
    #
    # o = dataManager._get_outside_data(now-datetime.timedelta(days=10), now)
    # o = dataManager._preprocess_outside_data(o.values())
    #
    # grouper = o.groupby([pd.Grouper(freq='1H')])
    # print(grouper['t_out'].mean())

    # zo = o.values()[0]
    # print zo.iloc[zo.shape[0]/2 - 5:]
    # print(o)
    # print(dataManager._preprocess_outside_data(o.values()))
    # o.dropna()

    # print("shape of outside data", o.shape)
    # print("number of 32 temperatures", (o["t_out"] == 32).sum())
    # t = dataManager.thermal_data(days_back=50)

    # print(t)

    # # plots the data here .
    # import matplotlib.pyplot as plt
    # z[0]["HVAC_Zone_Southzone"].plot()
    # plt.show()

    # zone_file = open("test_" + dm.building, 'wb')
    # pickle.dump(z, zone_file)
    # zone_file.close()

    # How we drop nan should be better documented in this code.

    # Look more carefully at data to see if we are learning weird stuff. In thermal model implement bias for the zones. This is important
    # because if two zones have the same temperature, but they differ in readings by 5 degrees, then there is no coefficient which can
    # make the temperature delta zero when they are actually same.

    # Update to make the timezone more friendly. for now everything in UTC time except the returned data.

    # TODO Make it such that i don't loose a minute in data preprocessing for every datapoint. could start at the
    # end of the interval each time by redoing the loop a bit. While instead of for loop? only counts for contigious
    # actions. If we have chnage of actions, it seems like we need to drop datapoints? Ask gabe for preprocessing and
    # if there is a better way to get the data since then we might not loose some values.
