import datetime
import json
import os
import pytz
from datetime import timedelta

import pandas as pd
import numpy as np
import requests
import yaml
from xbos import get_client
from xbos.services import mdal
from xbos.services.hod import HodClient


# TODO add energy data acquisition
# TODO FIX DAYLIGHT TIME CHANGE PROBLEMS


class ControllerDataManager:
    """
    # Class that handles all the data fetching and some of the preprocess for data that is relevant to controller
    and which does not have to be fetched every 15 min but only once. 
    """

    def __init__(self, controller_cfg, client,
                 now=datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))):

        self.controller_cfg = controller_cfg
        self.pytz_timezone = pytz.timezone(controller_cfg["Pytz_Timezone"])
        self.interval = controller_cfg["Interval_Length"]
        self.now = now.astimezone(self.pytz_timezone)

        self.client = client

        self.window_size = 1 # minutes. TODO should come from config. Has to be a multiple of 15 for weather getting.
        self.building = controller_cfg["Building"]
        self.hod_client = HodClient("xbos/hod", self.client)  # TODO hopefully i could incorporate this into the query.

    # def _preprocess_outside_data(self, outside_data):


    def _get_thermal_data(self, start, end):
        """Get thermostat status and temperature and outside temperature for thermal model.
        :param start: (datetime) time to start. relative to datamanager instance timezone.
        :param end: (datetime) time to end. relative to datamanager instance timezone.
        :return dict{zone: pd.df columns["tin", "a"]}, pd.df columns["tout"]. outside temperature has freq of 15 min and
        pd.df columns["tin", "a"] has freq of self.window_size. """

        # Converting start and end from datamanger timezone to UTC timezone.
        start = start.astimezone(pytz.timezone("UTC"))
        end = end.astimezone(pytz.timezone("UTC"))

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

        outside_temperature_query = """SELECT ?weather_station ?uuid FROM %s WHERE {
				?weather_station rdf:type brick:Weather_Temperature_Sensor.
				?weather_station bf:uuid ?uuid.
			};"""

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

        # get outside temperature data
        # TODO UPDATE OUTSIDE TEMPERATURE STUFF
        # TODO for now taking all weather stations and preprocessing it. Should be determined based on metadata.
        outside_temperature_query_data = self.hod_client.do_query(outside_temperature_query % self.building)["Rows"][0]

        # Get data from MDAL
        c = mdal.MDALClient("xbos/mdal", client=self.client)
        outside_temperature_data = c.do_query({
            'Composition': [outside_temperature_query_data["?uuid"]],
            # uuid from Mr.Plotter. should use outside_temperature_query_data["?uuid"],
            'Selectors': [mdal.MEAN]
            , 'Time': {'T0': start.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                       'T1': end.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                       'WindowSize': str(15) + 'min', # TODO document that we are getting 15 min intervals because then we get less nan values.
                       'Aligned': True}})


        outside_temperature_data = outside_temperature_data["df"]
        outside_temperature_data.columns = ["t_out"]

        # get the data for the thermostats for each zone.
        zone_thermal_data = {}
        for zone, dict in thermostat_query_data.items():
            # get the thermostat data
            dfs = c.do_query({'Composition': [dict["tstat_temperature"], dict["tstat_action"]],
                              'Selectors': [mdal.MEAN, mdal.MAX]
                                 , 'Time': {'T0': start.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                            'T1': end.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                            'WindowSize': str(self.window_size) + 'min',
                                            'Aligned': True}})
            df = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)

            zone_thermal_data[zone] = df.rename(columns={dict["tstat_temperature"]: 't_in', dict["tstat_action"]: 'a'})

        # TODO Note: The timezone for the data relies to be converted by MDAL to the local timezone.
        return zone_thermal_data, outside_temperature_data

    def _preprocess_thermal_data(self, zone_data, outside_data):
        """Preprocesses the data for the thermal model.
        :param zone_data: dict{zone: pd.df columns["tin", "a"]}
        :param outside_data: pd.df columns["tout"]. 
        NOTE: outside_data freq has to be a multiple of zone_data frequency and has to have a higher freq.
    
        :returns {zone: pd.df columns: t_in', 't_next', 'dt','t_out', 'action', 'a1', 'a2', [other mean zone temperatures]}
                 where t_out and zone temperatures are the mean values over the intervals. 
                 a1 is whether heating and a2 whether cooling."""

        # thermal data preprocess starts here
        # Heating
        def f1(row):
            """
            helper function to format the thermal model dataframe
            """
            if row['action'] == 1.:
                val = 1
            else:
                val = 0
            return val

        # if state is 2 we are doing cooling
        def f2(row):
            """
            helper function to format the thermal model dataframe
            """
            if row['action'] == 2.:
                val = 1
            else:
                val = 0
            return val

        def f3(row):
            """
            helper function to format the thermal model dataframe
            """
            if 0 < row['a'] <= 1:
                return 1
            elif 1 < row['a'] <= 2:
                return 2
            elif np.isnan(row['a']):
                return row['a']
            else:
                return 0

        all_temperatures = pd.concat([tstat_df["t_in"] for tstat_df in zone_data.values()], axis=1)
        all_temperatures.columns = ["zone_temperature_" + zone for zone in zone_data.keys()]
        zone_thermal_model_data = {}

        for zone in zone_data.keys():
            # Putting together outside and zone data.
            actions = zone_data[zone]["a"]
            thermal_model_data = pd.concat([all_temperatures, actions, outside_data],

                                           axis=1)  # should be copied data according to documentation
            thermal_model_data = thermal_model_data.rename(columns={"zone_temperature_" + zone: "t_in"})
            thermal_model_data['a'] = thermal_model_data.apply(f3, axis=1)

            # Assumption:
            # Outside temperature will have nan values because it does not have same frequency as zone data.
            # Hence, we fill with last known value to assume a constant temperature throughout intervals.
            thermal_model_data["t_out"] = thermal_model_data["t_out"].fillna(method="pad")

            # prepares final data type.
            thermal_model_data['change_of_action'] = (thermal_model_data['a'].diff(1) != 0).astype(
                'int').cumsum()  # given a row it's the number of times we have had an action change up till then. e.g. from nothing to heating.
            # This is accomplished by taking the difference of two consecutive rows and checking if their difference is 0 meaning that they had the same action.

            # following adds the fields "time", "dt" etc such that we accumulate all values where we have consecutively the same action.
            # maximally we group terms of total self.interval
            data_list = []
            for j in thermal_model_data.change_of_action.unique():
                for i in range(0, thermal_model_data[thermal_model_data['change_of_action'] == j].shape[0],
                               self.interval):
                    for dfs in [thermal_model_data[thermal_model_data['change_of_action'] == j][i:i + self.interval]]:
                        # we only look at intervals where the last and first value for T_in are not Nan.
                        dfs.dropna(subset=["t_in"])
                        zone_col_filter = ["zone_temperature_" in col for col in dfs.columns]
                        temp_data_dict = {'time': dfs.index[0],
                                          't_in': dfs['t_in'][0],
                                          't_next': dfs['t_in'][-1],
                                          # need to add windowsize for last timestep.
                                          'dt': (dfs.index[-1] - dfs.index[0]).seconds / 60 + self.window_size,
                                          't_out': dfs['t_out'].mean(),  # mean does not count Nan values
                                          'action': dfs['a'][0]}

                        for temperature_zone in dfs.columns[zone_col_filter]:
                            # mean does not count Nan values
                            temp_data_dict[temperature_zone] = dfs[temperature_zone].mean()
                        data_list.append(temp_data_dict)

            thermal_model_data = pd.DataFrame(data_list).set_index('time')
            thermal_model_data['a1'] = thermal_model_data.apply(f1, axis=1)
            thermal_model_data['a2'] = thermal_model_data.apply(f2, axis=1)

            thermal_model_data = thermal_model_data.dropna()  # final drop. Mostly if the whole interval for the zones or t_out were nan.

            zone_thermal_model_data[zone] = thermal_model_data

            print('one zone preproccessed')
        return zone_thermal_model_data

    def thermal_data(self, start=None, end=None, days_back=60):
        """
        :param start: In timezone of datamanger
        :param end: in timezone of datamanger
        :param if start is None, then we set start to end - timedelta(days=days_back). 
        :return: pd.df {zone: pd.df columns: t_in', 't_next', 'dt','t_out', 'action', 'a1', 'a2', [other mean zone temperatures]}
                 where t_out and zone temperatures are the mean values over the intervals. 
                 a1 is whether heating and a2 whether cooling.
        """
        if end is None:
            end = self.now
        if start is None:
            start = end - timedelta(days=days_back)
        z, o = self._get_thermal_data(start, end)
        return self._preprocess_thermal_data(z, o)



if __name__ == '__main__':

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
    import pickle

    with open("Freezing_avenal-recreation-center/2;HVAC_Zone_Large_Room;2018-06-03_07-58-08", "r") as f:
        thermal_data = pickle.load(f)["data"]["HVAC_Zone_Large_Room"]

    with open("Buildings/avenal-recreation-center/avenal-recreation-center.yml") as f:
        cfg = yaml.load(f)

    if cfg["Server"]:
        c = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        c = get_client()

    dataManager = ControllerDataManager(cfg, c)

    utc_zone = pytz.timezone("UTC")
    start = utc_zone.localize(thermal_data.index[0])
    end = utc_zone.localize(thermal_data.index[-1])
    dm_thermal_data, o = dataManager._get_thermal_data(start=start, end=end)

    print(o)

    print(dm_thermal_data)

    # # plots the data here .
    # import matplotlib.pyplot as plt
    # z[0]["HVAC_Zone_Southzone"].plot()
    # plt.show()

    # zone_file = open("test_" + dm.building, 'wb')
    # pickle.dump(z, zone_file)
    # zone_file.close()

