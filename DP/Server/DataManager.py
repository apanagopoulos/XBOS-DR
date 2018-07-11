import datetime
import json
import os
from datetime import timedelta

import pandas as pd
import numpy as np
import pytz
import requests
import yaml
from xbos import get_client
from xbos.services import mdal
from xbos.services.hod import HodClient

import utils


# TODO add energy data acquisition
# TODO FIX DAYLIGHT TIME CHANGE PROBLEMS

# util function. should be in util class.
def getDatetime(date_string):
    """Gets datetime from string with format HH:MM.
    :param date_string: string of format HH:MM
    :returns datetime.time() object with no associated timzone. """
    return datetime.datetime.strptime(date_string, "%H:%M").time()


def in_between(now, start, end):
    if start < end:
        return start <= now < end
    elif end < start:
        return start <= now or now < end
    else:
        return True


class DataManager:
    """
    # Class that handles all the data fetching and some of the preprocess
    """

    def __init__(self, controller_cfg, advise_cfg, client, zone,
                 now=None):
        """
        
        :param controller_cfg: 
        :param advise_cfg: 
        :param client: 
        :param zone: 
        :param now: in UTC time. If None (so no now is passed), take the current time. 
        """
        # TODO maybe get self.zone from config file.
        self.controller_cfg = controller_cfg
        self.advise_cfg = advise_cfg
        self.pytz_timezone = controller_cfg["Pytz_Timezone"]
        self.zone = zone
        self.interval = controller_cfg["Interval_Length"]
        if now is None:
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))
        self.now = now
        self.horizon = advise_cfg["Advise"]["MPCPredictiveHorizon"]
        self.c = client

    def preprocess_occ(self):
        """
        Returns the required dataframe for the occupancy predictions
        -------
        Pandas DataFrame
        """

        if self.advise_cfg["Advise"]["Occupancy_Sensors"]:
            hod = HodClient("xbos/hod",
                            self.c)  # TODO MAKE THIS WORK WITH FROM AND xbos/hod, FOR SOME REASON IT DOES NOT

            occ_query = """SELECT ?sensor ?uuid ?zone FROM %s WHERE {

						  ?sensor rdf:type brick:Occupancy_Sensor .
						  ?sensor bf:isPointOf/bf:isPartOf ?zone .
						  ?sensor bf:uuid ?uuid .
						  ?zone rdf:type brick:HVAC_Zone
						};
						""" % self.controller_cfg["Building"]  # get all the occupancy sensors uuids

            results = hod.do_query(occ_query)  # run the query
            uuids = [[x['?zone'], x['?uuid']] for x in results['Rows']]  # unpack

            # only choose the sensors for the zone specified in cfg
            query_list = []
            for i in uuids:
                if i[0] == self.zone:
                    query_list.append(i[1])

            # get the sensor data
            c = mdal.MDALClient("xbos/mdal", client=self.c)
            dfs = c.do_query({'Composition': query_list,
                              'Selectors': [mdal.MAX] * len(query_list),
                              'Time': {'T0': (self.now - timedelta(days=25)).strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                       'T1': self.now.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                       'WindowSize': str(self.interval) + 'min',
                                       'Aligned': True}})

            dfs = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)

            df = dfs[[query_list[0]]]
            df.columns.values[0] = 'occ'
            df.is_copy = False
            df.columns = ['occ']
            # perform OR on the data, if one sensor is activated, the whole zone is considered occupied
            for i in range(1, len(query_list)):
                df.loc[:, 'occ'] += dfs[query_list[i]]
            df.loc[:, 'occ'] = 1 * (df['occ'] > 0)

            return df.tz_localize(None)
        else:
            occupancy_array = self.advise_cfg["Advise"]["Occupancy"]

            now_time = self.now.astimezone(tz=pytz.timezone(self.controller_cfg["Pytz_Timezone"]))
            occupancy = []

            while now_time <= self.now + timedelta(hours=self.horizon):
                i = now_time.weekday()

                for j in occupancy_array[i]:
                    if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
                                  datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
                        occupancy.append(j[2])
                        break

                now_time += timedelta(minutes=self.interval)

            return occupancy

    def occupancy_archiver(self, start, end):

        hod = HodClient("xbos/hod",
                        self.c)  # TODO MAKE THIS WORK WITH FROM AND xbos/hod, FOR SOME REASON IT DOES NOT

        occ_query = """SELECT ?sensor ?uuid ?zone FROM %s WHERE {

                              ?sensor rdf:type brick:Occupancy_Sensor .
                              ?sensor bf:isPointOf/bf:isPartOf ?zone .
                              ?sensor bf:uuid ?uuid .
                              ?zone rdf:type brick:HVAC_Zone
                            };
                            """ % self.controller_cfg["Building"]  # get all the occupancy sensors uuids

        results = hod.do_query(occ_query)  # run the query
        uuids = [[x['?zone'], x['?uuid']] for x in results['Rows']]  # unpack

        # only choose the sensors for the zone specified in cfg
        query_list = []
        for i in uuids:
            if i[0] == self.zone:
                query_list.append(i[1])

        # get the sensor data
        c = mdal.MDALClient("xbos/mdal", client=self.c)
        dfs = c.do_query({'Composition': query_list,
                          'Selectors': [mdal.MAX] * len(query_list),
                          'Time': {'T0': start.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                   'T1': end.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                   'WindowSize': str(self.interval) + 'min',
                                   'Aligned': True}})

        dfs = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)

        df = dfs[[query_list[0]]]
        df.columns.values[0] = 'occ'
        df.is_copy = False
        df.columns = ['occ']
        # perform OR on the data, if one sensor is activated, the whole zone is considered occupied
        for i in range(1, len(query_list)):
            df.loc[:, 'occ'] += dfs[query_list[i]]
        df.loc[:, 'occ'] = 1 * (df['occ'] > 0)

        return df


    def better_occupancy_config(self, date):

        occupancy = self.advise_cfg["Advise"]["Occupancy"]

        weekday = date.weekday()

        date_occupancy = np.array(occupancy[weekday])

        start_date = date.replace(hour=0, minute=0, second=0)
        end_date = date.replace(day=date.day + 1, hour=0, minute=0, second=0)
        date_range = pd.date_range(start=start_date, end=end_date, freq="1T")

        def combine_date_time(time, date):
            datetime_time = utils.get_time_datetime(time)
            return date.replace(hour=datetime_time.hour, minute=datetime_time.minute, second=0)

        # create nan filled dataframe and populate it
        df_occupancy = pd.DataFrame(columns=["occ"], index=date_range)

        for interval_occupancy in date_occupancy:
            start, end, occ = interval_occupancy
            occ = float(occ)
            start = combine_date_time(start, date)
            end = combine_date_time(end, date)
            # if we are going into the next day.
            if end <= start and end.hour == 0 and end.minute == 0:
                end = end.replace(day=end.day + 1)
            df_occupancy.loc[start:end, "occ"] = occ
        return df_occupancy


    def weather_fetch(self, fetch_attempts=10):

        from dateutil import parser
        file_name = "./weather/weather_" + self.zone + "_" + self.controller_cfg["Building"] + ".json"

        coordinates = self.controller_cfg["Coordinates"]

        weather_fetch_successful = False
        while not weather_fetch_successful and fetch_attempts > 0:
            if not os.path.exists(file_name):
                temp = requests.get("https://api.weather.gov/points/" + coordinates).json()
                weather = requests.get(temp["properties"]["forecastHourly"])
                data = weather.json()
                with open(file_name, 'wb') as f:
                    json.dump(data, f)

            try:
                with open(file_name, 'r') as f:
                    myweather = json.load(f)
                    weather_fetch_successful = True
            except:
                # Error with reading json file. Refetching data.
                print("Warning, received bad weather.json file. Refetching from archiver.")
                os.remove(file_name)
                weather_fetch_successful = False
                fetch_attempts -= 1

        if fetch_attempts == 0:
            raise Exception("ERROR, Could not get good data from weather service.")

        # got an error on parse in the next line that properties doesnt exit
        json_start = parser.parse(myweather["properties"]["periods"][0]["startTime"])
        if (json_start.hour < self.now.astimezone(tz=pytz.timezone(self.pytz_timezone)).hour) or \
                (datetime.datetime(json_start.year, json_start.month, json_start.day).replace(
                    tzinfo=pytz.timezone(self.pytz_timezone)) <
                     datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(
                         tz=pytz.timezone(self.pytz_timezone))):
            temp = requests.get("https://api.weather.gov/points/" + coordinates).json()
            weather = requests.get(temp["properties"]["forecastHourly"])
            data = weather.json()
            with open(file_name, 'w') as f:
                json.dump(data, f)
            myweather = json.load(open(file_name))

        weather_predictions = {}

        for i, data in enumerate(myweather["properties"]["periods"]):
            hour = parser.parse(data["startTime"]).hour
            weather_predictions[hour] = int(data["temperature"])
            if i == self.horizon:
                break

        return weather_predictions

    def thermostat_setpoints(self, start, end):
        """Gets setpoints"""

        WINDOW_SIZE = 1 # min for getting data from archiver.

        cooling_setpoint_query = """SELECT ?zone ?uuid FROM %s WHERE {
                    ?tstat rdf:type brick:Thermostat .
                    ?tstat bf:controls ?RTU .
                    ?RTU rdf:type brick:RTU .
                    ?RTU bf:feeds ?zone. 
                    ?zone rdf:type brick:HVAC_Zone .
                    ?tstat bf:hasPoint ?setpoint .
                    ?setpoint rdf:type brick:Supply_Air_Temperature_Cooling_Setpoint .
                    ?setpoint bf:uuid ?uuid.
                    };"""

        heating_setpoint_query = """SELECT ?zone ?uuid FROM %s WHERE {
                    ?tstat rdf:type brick:Thermostat .
                    ?tstat bf:controls ?RTU .
                    ?RTU rdf:type brick:RTU .
                    ?RTU bf:feeds ?zone. 
                    ?zone rdf:type brick:HVAC_Zone .
                    ?tstat bf:hasPoint ?setpoint .
                    ?setpoint rdf:type brick:Supply_Air_Temperature_Heating_Setpoint .
                    ?setpoint bf:uuid ?uuid.
                    };"""


        hod_client = HodClient("xbos/hod", self.c)


        # get query data
        temp_thermostat_query_data = {
            "cooling_setpoint": hod_client.do_query(cooling_setpoint_query % self.controller_cfg["Building"])["Rows"],
            "heating_setpoint": hod_client.do_query(heating_setpoint_query % self.controller_cfg["Building"])["Rows"],
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
        mdal_client = mdal.MDALClient("xbos/mdal", client=self.c)
        zone_setpoint_data = {}
        for zone, dict in thermostat_query_data.items():

            mdal_query = {'Composition': [dict["heating_setpoint"], dict["cooling_setpoint"]],
                                        'Selectors': [mdal.MEAN, mdal.MEAN]
                                           , 'Time': {'T0': start.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                                      'T1': end.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                                      'WindowSize': str(WINDOW_SIZE) + 'min',
                                                      'Aligned': True}}

            # get the thermostat data
            df = utils.get_mdal_data(mdal_client, mdal_query)
            zone_setpoint_data[zone] = df.rename(columns={dict["heating_setpoint"]: 't_low',
                                                          dict["cooling_setpoint"]: 't_high'})

        return zone_setpoint_data

        #
        # uuids = [self.advise_cfg["Data_Manager"]["UUIDS"]['Thermostat_high'],
        #          self.advise_cfg["Data_Manager"]["UUIDS"]['Thermostat_low'],
        #          self.advise_cfg["Data_Manager"]["UUIDS"]['Thermostat_mode']]
        #
        # c = mdal.MDALClient("xbos/mdal", client=self.c)
        # dfs = c.do_query({'Composition': uuids,
        #                   'Selectors': [mdal.MEAN, mdal.MEAN, mdal.MEAN],
        #                   'Time': {'T0': (self.now - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
        #                            'T1': self.now.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
        #                            'WindowSize': '1min',
        #                            'Aligned': True}})
        #
        # df = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)
        # df = df.rename(columns={uuids[0]: 'T_High', uuids[1]: 'T_Low', uuids[2]: 'T_Mode'})
        #
        # return df['T_High'][-1], df['T_Low'][-1], df['T_Mode'][-1]

    def prices(self):

        price_array = self.controller_cfg["Pricing"][self.controller_cfg["Pricing"]["Energy_Rates"]]

        def in_between(now, start, end):
            if start < end:
                return start <= now < end
            elif end < start:
                return start <= now or now < end
            else:
                return True

        if self.controller_cfg["Pricing"]["Energy_Rates"] == "Server":
            # not implemented yet, needs fixing from the archiver
            # (always says 0, problem unless energy its free and noone informed me)
            raise ValueError('SERVER MODE IS NOT YET IMPLEMENTED FOR ENERGY PRICING')
        else:
            now_time = self.now.astimezone(tz=pytz.timezone(self.controller_cfg["Pytz_Timezone"]))
            pricing = []

            DR_start_time = [int(self.controller_cfg["Pricing"]["DR_Start"].split(":")[0]),
                             int(self.controller_cfg["Pricing"]["DR_Start"].split(":")[1])]
            DR_finish_time = [int(self.controller_cfg["Pricing"]["DR_Finish"].split(":")[0]),
                              int(self.controller_cfg["Pricing"]["DR_Finish"].split(":")[1])]

            while now_time <= self.now + timedelta(hours=self.horizon):
                i = 1 if now_time.weekday() >= 5 or self.controller_cfg["Pricing"]["Holiday"] else 0
                for j in price_array[i]:
                    if in_between(now_time.time(), datetime.time(DR_start_time[0], DR_start_time[1]),
                                  datetime.time(DR_finish_time[0], DR_finish_time[1])) and \
                            (self.controller_cfg["Pricing"][
                                 "DR"]):  # TODO REMOVE ALLWAYS HAVING DR ON FRIDAY WHEN DR SUBSCRIBE IS IMPLEMENTED
                        pricing.append(self.controller_cfg["Pricing"]["DR_Price"])
                    elif in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
                                    datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
                        pricing.append(j[2])
                        break

                now_time += timedelta(minutes=self.interval)

        return pricing

    def better_prices(self, date):
        """For the whole day."""

        price_array = self.controller_cfg["Pricing"][self.controller_cfg["Pricing"]["Energy_Rates"]]


        if self.controller_cfg["Pricing"]["Energy_Rates"] == "Server":
            # not implemented yet, needs fixing from the archiver
            # (always says 0, problem unless energy its free and noone informed me)
            raise ValueError('SERVER MODE IS NOT YET IMPLEMENTED FOR ENERGY PRICING')
        else:

            DR_start_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Start"])
            DR_finish_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Finish"])
            DR_price = self.controller_cfg["Pricing"]["DR_Price"]
            is_DR = self.controller_cfg["Pricing"]["DR"]


            is_holiday_weekend = date.weekday() >= 5 or self.controller_cfg["Pricing"]["Holiday"]

            day_type = int(is_holiday_weekend)

            date_prices = np.array(price_array[day_type])

            start_date = date.replace(hour=0, minute=0, second=0)
            end_date = date.replace(day=date.day + 1, hour=0, minute=0, second=0)
            date_range = pd.date_range(start=start_date, end=end_date, freq="1T")

            def combine_date_time(time, date):
                datetime_time = utils.get_time_datetime(time)
                return date.replace(hour=datetime_time.hour, minute=datetime_time.minute, second=0)

            # create nan filled dataframe and populate it
            df_prices = pd.DataFrame(columns=["price"], index=date_range)

            for interval_price in date_prices:
                start, end, price = interval_price
                price = float(price)
                start = combine_date_time(start, date)
                end = combine_date_time(end, date)
                # if we are going into the next day.
                if end <= start and end.hour == 0 and end.minute == 0:
                    end = end.replace(day=end.day + 1)
                df_prices.loc[start:end] = price

            # setting dr prices.
            if is_DR:
                df_prices.loc[DR_start_time:DR_finish_time] = DR_price


        return df_prices

    def better_safety(self, date):
        # for whole day.

        setpoints_array = self.advise_cfg["Advise"]["SafetySetpoints"]

        weekday = date.weekday()

        date_setpoints = np.array(setpoints_array[weekday])

        start_date = date.replace(hour=0, minute=0, second=0)
        end_date = date.replace(day=date.day + 1, hour=0, minute=0, second=0)
        date_range = pd.date_range(start=start_date, end=end_date, freq="1T")

        def combine_date_time(time, date):
            datetime_time = utils.get_time_datetime(time)
            return date.replace(hour=datetime_time.hour, minute=datetime_time.minute, second=0)

        # create nan filled dataframe and populate it
        df_setpoints = pd.DataFrame(columns=["t_high", "t_low"], index=date_range)

        for interval_setpoints in date_setpoints:
            start, end, t_low, t_high = interval_setpoints
            start = combine_date_time(start, date)
            end = combine_date_time(end, date)
            # if we are going into the next day.
            if end <= start and end.hour == 0 and end.minute == 0:
                end = end.replace(day=end.day + 1)
            if t_low is None or t_high is None or t_low == "None" or t_high == "None":
                raise Exception("Safety should have no None.")
            else:
                t_low = float(t_low)
                t_high = float(t_high)
                df_setpoints.loc[start:end, "t_high"] = t_high
                df_setpoints.loc[start:end, "t_low"] = t_low

        return df_setpoints

    def better_comfortband(self, date):
        # for whole day.

        setpoints_array = self.advise_cfg["Advise"]["Comfortband"]
        df_safety = self.better_safety(date)

        weekday = date.weekday()


        date_setpoints = np.array(setpoints_array[weekday])

        start_date = date.replace(hour=0, minute=0, second=0)
        end_date = date.replace(day=date.day + 1, hour=0, minute=0, second=0)
        date_range = pd.date_range(start=start_date, end=end_date, freq="1T")

        def combine_date_time(time, date):
            datetime_time = utils.get_time_datetime(time)
            return date.replace(hour=datetime_time.hour, minute=datetime_time.minute, second=0)

        # create nan filled dataframe and populate it
        df_setpoints = pd.DataFrame(columns=["t_high", "t_low"], index=date_range)

        for interval_setpoints in date_setpoints:
            start, end, t_low, t_high = interval_setpoints
            start = combine_date_time(start, date)
            end = combine_date_time(end, date)
            # if we are going into the next day.
            if end <= start and end.hour == 0 and end.minute == 0:
                end = end.replace(day=end.day + 1)
            if t_low is None or t_high is None or t_low == "None" or t_high == "None":
                interval_safety = df_safety[start:end]
                df_setpoints[start:end] = interval_safety
            else:
                t_low = float(t_low)
                t_high = float(t_high)
                df_setpoints.loc[start:end, "t_high"] = t_high
                df_setpoints.loc[start:end, "t_low"] = t_low

        return df_setpoints


    def building_setpoints(self):

        setpoints_array = self.advise_cfg["Advise"]["Comfortband"]
        safety_temperatures = self.advise_cfg["Advise"]["SafetySetpoints"]

        now_time = self.now.astimezone(tz=pytz.timezone(self.controller_cfg["Pytz_Timezone"]))
        setpoints = []

        while now_time <= self.now + timedelta(hours=self.horizon):
            weekday = now_time.weekday()

            for j in setpoints_array[weekday]:
                if in_between(now_time.time(), getDatetime(j[0]), getDatetime(j[1])) and \
                        (j[2] != "None" or j[3] != "None"): # TODO come up with better None value detection.
                    setpoints.append([j[2], j[3]])
                    break

                # if we have none values, replace the values with safetytemperatures.
                elif in_between(now_time.time(), getDatetime(j[0]), getDatetime(j[1])) and \
                        (j[2] == "None" or j[3] == "None"):
                    for safety_temperature_time in safety_temperatures[weekday]:
                        if in_between(now_time.time(), getDatetime(safety_temperature_time[0]),
                                      getDatetime(safety_temperature_time[1])):
                            setpoints.append([safety_temperature_time[2], safety_temperature_time[3]])
                            break

            now_time += timedelta(minutes=self.interval)

        return setpoints

    def safety_constraints(self):
        setpoints_array = self.advise_cfg["Advise"]["SafetySetpoints"]

        def in_between(now, start, end):
            if start < end:
                return start <= now < end
            elif end < start:
                return start <= now or now < end
            else:
                return True

        now_time = self.now.astimezone(tz=pytz.timezone(self.controller_cfg["Pytz_Timezone"]))
        setpoints = []

        while now_time <= self.now + timedelta(hours=self.horizon):
            i = now_time.weekday()

            for j in setpoints_array[i]:
                if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
                              datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
                    setpoints.append([j[2], j[3]])
                    break

            now_time += timedelta(minutes=self.interval)

        return setpoints


if __name__ == '__main__':
    # Test for ciee
    with open("Buildings/ciee/ciee.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    with open("Buildings/" + cfg["Building"] + "/ZoneConfigs/HVAC_Zone_Centralzone.yml", 'r') as ymlfile:
        advise_cfg = yaml.load(ymlfile)

    if cfg["Server"]:
        c = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        c = get_client()

    dm = DataManager(cfg, advise_cfg, c, "HVAC_Zone_Centralzone")

    print "Weather Predictions:"
    # print dm.weather_fetch()
    # print "Occupancy Data"
    print dm.preprocess_occ()
    # print "Thermostat Setpoints:"
    # print dm.thermostat_setpoints()
    # print "Prices:"
    # print dm.prices()
    # print "Setpoints for each future interval:"
    # print dm.building_setpoints()
    # print "Safety constraints for each future interval:"
    # print dm.safety_constraints()
