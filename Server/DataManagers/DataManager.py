import datetime
import json
from datetime import timedelta

import numpy as np
import os
import pandas as pd
import pytz
import requests
from xbos.services import mdal
from xbos.services.hod import HodClient

import utils


# TODO add energy data acquisition
# TODO FIX DAYLIGHT TIME CHANGE PROBLEMS

# TODO NOTE: The better_ datemethods all take the end inclusive from the config files.

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
        self.pytz_timezone = pytz.timezone(controller_cfg["Pytz_Timezone"])
        self.zone = zone
        self.interval = controller_cfg["Interval_Length"]
        if now is None:
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))
        self.now = now
        self.horizon = advise_cfg["Advise"]["MPCPredictiveHorizon"]
        self.c = client

    # TODO NEED TO BE REPLACED by better occupancy data methods.
    @property
    def preprocess_occ_mdal(self):
        """
        Returns the required dataframe for the occupancy predictions
        -------
        Pandas DataFrame
        """

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

    def preprocess_occ_cfg(self):

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

    def get_better_occupancy_config(self, date, freq="1T"):
        """
        Gets the occupancy from the zone configuration file. Uses the provided date for which the prices should
        hold. Cannot have Nan values.
        :param date: The date for which we want to get the occupancy from config. Timezone aware.
        :param freq: The frequency of time series. Default is one minute.
        :return: pd.df columns="occ" with time_series index for the date provided and in timezone aware
         datetime as provided by the configuration file. With freq as frequency. 
        """
        # Set the date to the controller timezone.
        date = date.astimezone(tz=pytz.timezone(self.controller_cfg["Pytz_Timezone"]))

        occupancy = self.advise_cfg["Advise"]["Occupancy"]

        weekday = date.weekday()

        date_occupancy = np.array(occupancy[weekday])

        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + datetime.timedelta(days=1)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # create nan filled dataframe and populate it
        df_occupancy = pd.DataFrame(columns=["occ"], index=date_range)

        for interval_occupancy in date_occupancy:
            start, end, occ = interval_occupancy
            start = utils.combine_date_time(start, date)
            end = utils.combine_date_time(end, date)
            # if we are going into the next day.
            if end <= start and end.hour == 0 and end.minute == 0:
                end += datetime.timedelta(days=1)
            if occ is None or occ == "None":
                raise Exception("Config Occupancy should have no None.")
            df_occupancy.loc[start:end, "occ"] = float(occ)
        return df_occupancy





    def thermostat_setpoints(self, start, end, window_size=1):
        """
        Gets the thermostat setpoints from archiver from start to end. Does not preprocess the data.
        :param start: datetime in utc time.
        :param end: datetime in utc time.
        :param window_size: The frequency with which to get the data.
        :return: pd.df columns="t_high", "t_low" with timeseries in utc time with freq=window_size. 
        """

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
                           'WindowSize': str(window_size) + 'min',
                           'Aligned': True}}

            # get the thermostat data
            df = utils.get_mdal_data(mdal_client, mdal_query)
            zone_setpoint_data[zone] = df.rename(columns={dict["heating_setpoint"]: 't_low',
                                                          dict["cooling_setpoint"]: 't_high'})

        return zone_setpoint_data

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

    def get_better_prices(self, date, freq="1T"):
        """
        Gets the prices from the building configuration file. Uses the provided date for which the prices should
        hold. Cannot have Nan values.
        :param date: The date for which we want to get the prices from config. Timezone aware.
        :param freq: The frequency of time series. Default is one minute.
        :return: pd.df columns="price" with time_series index for the date provided and in timezone aware
         datetime as provided by the configuration file. 
        """
        # Set the date to the controller timezone.
        date = date.astimezone(tz=self.pytz_timezone)

        price_array = self.controller_cfg["Pricing"][self.controller_cfg["Pricing"]["Energy_Rates"]]

        if self.controller_cfg["Pricing"]["Energy_Rates"] == "Server":
            # not implemented yet, needs fixing from the archiver
            # (always says 0, problem unless energy its free and noone informed me)
            raise ValueError('SERVER MODE IS NOT YET IMPLEMENTED FOR ENERGY PRICING')
        else:
            # Get DR Data and whether to get DR prices.
            DR_start_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Start"])
            DR_start = datetime.datetime.combine(date, DR_start_time)
            DR_finish_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Finish"])
            DR_finish = datetime.datetime.combine(date, DR_finish_time)
            DR_price = self.controller_cfg["Pricing"]["DR_Price"]
            is_DR = self.controller_cfg["Pricing"]["DR"]

            # The type of pricing to choose as integer because that's how we index in the config file.
            is_holiday_weekend = date.weekday() >= 5 or self.controller_cfg["Pricing"]["Holiday"]
            day_type = int(is_holiday_weekend)

            # Get prices from config as array.
            date_prices = np.array(price_array[day_type])

            # The start and end day for the date for which to get prices.
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + datetime.timedelta(days=1)
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

            # create nan filled dataframe and then populate it
            df_prices = pd.DataFrame(columns=["price"], index=date_range)

            for interval_price in date_prices:
                start, end, price = interval_price
                start = utils.combine_date_time(start, date)
                end = utils.combine_date_time(end, date)
                # if we are going into the next day.
                if end <= start and end.hour == 0 and end.minute == 0:
                    end += datetime.timedelta(days=1)
                if price is None or price == "None":
                    raise Exception("Prices should have no None.")
                df_prices.loc[start:end] = float(price)

            # setting dr prices.
            if is_DR:
                if DR_price is None or DR_price == "None":
                    raise Exception("DR Prices should have no None.")
                df_prices.loc[DR_start:DR_finish] = float(DR_price)
            df_prices = df_prices.astype(np.float)

            return df_prices

    def get_better_lambda(self, date, freq="1T"):
        """
        Gets the lambdas from the building configuration file. Uses the provided date for which the lambdas should
        hold. Cannot have Nan values.
        :param date: The date for which we want to get the data from config. Timezone aware.
        :param freq: The frequency of time series. Default is one minute.
        :return: pd.df columns="lambda" with time_series index for the date provided and in timezone aware
         datetime as provided by the configuration file. 
        """
        # Set the date to the controller timezone.
        date = date.astimezone(tz=self.pytz_timezone)

        # Get lambdas
        general_lambda = self.advise_cfg["Advise"]["General_Lambda"]
        dr_lambda = self.advise_cfg["Advise"]["DR_Lambda"]

        # Get DR Data and whether to get DR prices.
        DR_start_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Start"])
        DR_start = datetime.datetime.combine(date, DR_start_time)
        DR_finish_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Finish"])
        DR_finish = datetime.datetime.combine(date, DR_finish_time)
        is_DR = self.controller_cfg["Pricing"]["DR"]

        # The start and end day for the date for which to get prices.
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + datetime.timedelta(days=1)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # create nan filled dataframe and then populate it
        df_lambda = pd.DataFrame(columns=["lambda"], index=date_range)

        if general_lambda is None or general_lambda == "None":
            raise Exception("General Lambda should not be None.")

        df_lambda.loc[start_date:end_date] = float(general_lambda)

        # setting dr lambda.
        if is_DR:
            if dr_lambda is None or dr_lambda == "None":
                raise Exception("DR Lambda should not be None if we have a DR event.")
            df_lambda.loc[DR_start:DR_finish] = float(dr_lambda)

        return df_lambda

    def get_better_is_dr(self, date, freq="1T"):
        """
        Gets the is_dr from the building configuration file. Uses the provided date for which the lambdas should
        hold. Cannot have Nan values.
        :param date: The date for which we want to get the data from config. Timezone aware.
        :param freq: The frequency of time series. Default is one minute.
        :return: pd.series with time_series index for the date provided and in timezone aware
         datetime as provided by the configuration file. 
        """
        # Set the date to the controller timezone.
        date = date.astimezone(tz=self.pytz_timezone)

        # Get DR Data and whether to get DR prices.
        DR_start_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Start"])
        DR_start = datetime.datetime.combine(date, DR_start_time)
        DR_finish_time = utils.get_time_datetime(self.controller_cfg["Pricing"]["DR_Finish"])
        DR_finish = datetime.datetime.combine(date, DR_finish_time)
        is_DR = self.controller_cfg["Pricing"]["DR"]

        # The start and end day for the date for which to get prices.
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + datetime.timedelta(days=1)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # create nan filled dataframe and then populate it
        df_is_dr = pd.Series(index=date_range)

        df_is_dr.loc[start_date:end_date] = False

        # setting if Dr time.
        if is_DR:
            df_is_dr.loc[DR_start:DR_finish] = True

        return df_is_dr

    def get_better_safety(self, date, freq="1T"):
        """
        Gets the safety_setpoints from the zone configuration file. Uses the provided date for which the setpoints should
        hold. Cannot have Nan values.
        :param date: The date for which we want to get the safety setpoints from config. Timezone aware.
        :param freq: The frequency of time series. Default is one minute.
        :return: pd.df columns=t_high, t_low with time_series index for the date provided and in timezone aware
         datetime as provided by the configuration file. 
        """
        # Set the date to the controller timezone.
        date = date.astimezone(tz=self.pytz_timezone)

        setpoints_array = self.advise_cfg["Advise"]["SafetySetpoints"]

        weekday = date.weekday()

        date_setpoints = np.array(setpoints_array[weekday])

        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + datetime.timedelta(days=1)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # create nan filled dataframe and populate it
        df_setpoints = pd.DataFrame(columns=["t_high", "t_low"], index=date_range)

        for interval_setpoints in date_setpoints:
            start, end, t_low, t_high = interval_setpoints
            start = utils.combine_date_time(start, date)
            end = utils.combine_date_time(end, date)
            # if we are going into the next day.
            if end <= start and end.hour == 0 and end.minute == 0:
                end += datetime.timedelta(days=1)
            if t_low is None or t_high is None or t_low == "None" or t_high == "None":
                raise Exception("Safety should have no None.")
            else:
                t_low = float(t_low)
                t_high = float(t_high)
                df_setpoints.loc[start:end, "t_high"] = t_high
                df_setpoints.loc[start:end, "t_low"] = t_low

        return df_setpoints

    def get_better_comfortband(self, date, freq="1T"):
        """
        Gets the comfortband from the zone configuration file. Uses the provided date for which the comfortband should
        hold. If there are nan's in the comfortband, then those will be replaced with the saftysetpoints.
        :param date: The date for which we want to get the comfortband from config. Timezone aware.
        :param freq: The frequency of time series. Default is one minute.
        :return: pd.df columns=t_high, t_low with time_series index for the date provided and in timezone aware
         datetime as provided by the configuration file.
        """
        # Set the date to the controller timezone.
        date = date.astimezone(tz=self.pytz_timezone)

        setpoints_array = self.advise_cfg["Advise"]["Comfortband"]
        df_safety = self.get_better_safety(date, freq)

        weekday = date.weekday()

        date_setpoints = np.array(setpoints_array[weekday])

        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + datetime.timedelta(days=1)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # create nan filled dataframe and populate it
        df_setpoints = pd.DataFrame(columns=["t_high", "t_low"], index=date_range)

        for interval_setpoints in date_setpoints:
            start, end, t_low, t_high = interval_setpoints
            start = utils.combine_date_time(start, date)
            end = utils.combine_date_time(end, date)
            # if we are going into the next day.
            if end <= start and end.hour == 0 and end.minute == 0:
                end += datetime.timedelta(days=1)
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

        now_time = self.now.astimezone(tz=self.pytz_timezone)
        setpoints = []

        while now_time <= self.now + timedelta(hours=self.horizon):
            weekday = now_time.weekday()

            for j in setpoints_array[weekday]:
                if in_between(now_time.time(), getDatetime(j[0]), getDatetime(j[1])) and \
                        (j[2] != "None" or j[3] != "None"):  # TODO come up with better None value detection.
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
    building = "ciee"
    zone = "HVAC_Zone_Centralzone"
    cfg = utils.get_config(building)
    advise_cfg = utils.get_zone_config(building, zone)

    client = utils.choose_client()

    dm = DataManager(cfg, advise_cfg, None, "HVAC_Zone_Centralzone")

    start = utils.get_utc_now()
    end = start + datetime.timedelta(days=1)

    print(dm.weather_fetch(start, end, 15))

    # print "Weather Predictions:"
    # print dm.weather_fetch()
    # print "Occupancy Data"
    # print dm.preprocess_occ()
    # print "Thermostat Setpoints:"
    # print dm.thermostat_setpoints()
    # print "Prices:"
    # print dm.prices()
    # print "Setpoints for each future interval:"
    # print dm.building_setpoints()
    # print "Safety constraints for each future interval:"
    # print dm.safety_constraints()
