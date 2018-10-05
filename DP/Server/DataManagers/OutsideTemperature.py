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

import requests
import os

from dateutil import parser

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.dirname(FILE_PATH) # The Server is Parent for now.


def weather_fetch(building, start, end, interval, fetch_attempts=10):
    """Gets the weather predictions from weather.gov
    :param start: (datetime timezone_aware) when in the future the predictions should start
    :param end: (datetime timezone_aware) when the weather fetch should end. inclusive
    :param interval: frequency of data in minutes.
    :param fetch_attempts: (int) number of attempts we should try to get weather from weather.gov
    :return pd.series with date range from start to end in freq=interval minutes"""

    cfg_building = utils.get_config(building)
    pytz_tz = pytz.timezone(cfg_building["Pytz_Timezone"])


    now = utils.get_utc_now()
    now_cfg_timezone = now.astimezone(tz=pytz_tz)

    start_cfg_tz = start.astimezone(tz=pytz_tz)
    end_cfg_tz = end.astimezone(tz=pytz_tz)

    # we should get about 6 days of data with each weather fetch i think.
    assert start_cfg_tz <= end_cfg_tz <= now + datetime.timedelta(days=6)

    # Try to get data from weather data that was saved.
    file_name = SERVER_PATH + "/weather/weather_" + building + ".pkl"
    if os.path.exists(file_name):
        try:
            weather_data = pd.read_pickle(file_name)
            # Checking if the start and end times are in the data range and also ensuring that our data is not more
            # than a day old.
            if weather_data.index[0] <= start_cfg_tz <= end_cfg_tz <= weather_data.index[-1] \
                    and weather_data.index[0] <= now_cfg_timezone + datetime.timedelta(days=1):
                return utils.smart_resample(weather_data, start_cfg_tz, end_cfg_tz, interval, method="interpolate")
        except:
            os.remove(file_name)

    # attempt to fetch weather from weather.gove
    try_counter = 0
    weather_fetch_successful = False
    coordinates = cfg_building["Coordinates"]
    while not weather_fetch_successful and try_counter <= fetch_attempts:
        try:
            weather_meta = requests.get("https://api.weather.gov/points/" + coordinates).json()
            weather_json = requests.get(weather_meta["properties"]["forecastHourly"])
            weather_data_dictionary = weather_json.json()

            weather_fetch_successful = True
        except:
            try_counter += 1

    # If we couldn't get data, then we need to raise an exception.
    if not weather_fetch_successful:
        raise Exception("ERROR, Could not get good data from weather service.")

    # convert the data to a pandas Series.
    weather_times = []
    weather_temperatures = []
    weather_temperature_unit = []
    for row in weather_data_dictionary["properties"]["periods"]:
        weather_times.append(parser.parse(row["startTime"]))
        weather_temperatures.append(row["temperature"])
        weather_temperature_unit.append(row["temperatureUnit"])
        if weather_temperature_unit[-1] != "F":
            raise Exception(
                "Weather fetch got data which was not Fahrenheit. It had units: %s" % weather_temperature_unit[-1])

    weather_data = pd.Series(data=weather_temperatures, index=weather_times)

    # store the data
    weather_data.to_pickle(file_name)

    # return data interpolated to have it start and end at the given times.
    return utils.smart_resample(weather_data, start_cfg_tz, end_cfg_tz, interval, method="interpolate")


def _preprocess_mdal_outside_data(outside_data):
    """
    :param outside_data: pd.df with column for each weather station.
    :return: pd.Series
    """

    # since the archiver had a bug while storing weather.gov data, nan values were stored as 32. We will
    # replace any 32 values with Nan. Since we usually have more than one weather station, we can then take the
    # average across those to compensate. If no other weather stations, then issue a warning that
    # we might be getting rid of real data and we won't be able to recover through the mean.
    if len(outside_data) == 1:
        print("WARNING: Only one weather station for selected region. We need to replace 32 values with Nan due to "
              "past inconsistencies, but not enough data to compensate for the lost data by taking mean.")

    outside_data = outside_data.applymap(
            lambda t: np.nan if t == 32 else t)  # TODO this only works for fahrenheit now.

    # Note: Assuming same index for all weather station data returned by mdal
    outside_data = outside_data.mean(axis=1)

    # outside temperature may contain nan values.
    # Hence, we interpolate values linearly,
    # since outside temperature does not need to be as accurate as inside data.
    final_outside_data = outside_data.interpolate()

    return final_outside_data

def _get_mdal_outside_data(building, start, end):
    """Get outside temperature for thermal model.
    :param start: (datetime) time to start. timezone aware
    :param end: (datetime) time to end. timezone aware
    :param inclusive: (bool) whether the end time should be inclusive.
                    which means that we get the time for the end, plus 15 min.
    :return ({uuid: (pd.df) (col: "t_out) outside_data})  outside temperature has freq of 15 min and
    pd.df columns["tin", "action"] has freq of window_size. """
    assert end <= utils.get_utc_now() # we can only get historic data.

    client = get_client()
    hod_client = HodClient("xbos/hod", client)


    outside_temperature_query = """SELECT ?weather_station ?uuid FROM %s WHERE {
                                ?weather_station rdf:type brick:Weather_Temperature_Sensor.
                                ?weather_station bf:uuid ?uuid.
                                };"""

    # get outside temperature data
    # TODO for now taking all weather stations and preprocessing it. Should be determined based on metadata.
    outside_temperature_query_data = hod_client.do_query(outside_temperature_query % building)["Rows"]
    all_uuids =  [row["?uuid"] for row in outside_temperature_query_data]

    # Get data from MDAL
    mdal_client = mdal.MDALClient("xbos/mdal", client=client)
    mdal_query = {
        'Composition': all_uuids,
        'Selectors': [mdal.MEAN] * len(all_uuids)
        , 'Time': {'T0': start.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                   'T1': end.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                   'WindowSize': str(15) + 'min', # we are getting 15 min intervals because then we get fewer nan values. MDAL specific issue.
                   'Aligned': True}}

    mdal_outside_data = utils.get_mdal_data(mdal_client, mdal_query)
    return mdal_outside_data


def get_historic_outside_data(building, start, end, interval):
    # TODO what to do if all values are nan?
    mdal_data = _get_mdal_outside_data(building, start, end)
    preprocessed_data = _preprocess_mdal_outside_data(mdal_data)
    return utils.smart_resample(preprocessed_data, start, end, interval, method="interpolate")


def get_outside_temperatures(building, start, end, interval=15):
    """
    Get outside weather from start to end. Will combine historic and weather predictions data when necessary.
    :param start: datetime timezone aware
    :param end: datetime timezone aware
    :param interval: (int minutes) interval of the returned timeseries in minutes.
    :return: pd.Series with combined data of historic and prediction outside weather.
    """
    # For finding out if start or/and end are before or after the current time.
    utc_now = utils.get_utc_now()

    cfg_building = utils.get_config(building)

    # Set start and end to correct timezones
    cfg_timezone = utils.get_config_timezone(cfg_building)
    start_utc = start.astimezone(tz=pytz.utc)
    end_utc = end.astimezone(tz=pytz.utc)
    start_cfg_timezone = start.astimezone(tz=cfg_timezone)
    end_cfg_timezone = end.astimezone(tz=cfg_timezone)
    now_cfg_timezone = utc_now.astimezone(tz=cfg_timezone)

    # Getting temperatures.
    # Note, adding interval minute intervals to ensure that we get at least 1 interval from historic/future

    # Populating the outside_temperatures pd.Series for MPC use. Ouput is in cfg timezone.
    outside_temperatures = pd.Series(index=pd.date_range(start_cfg_timezone, end_cfg_timezone, freq=str(interval)+"T"))
    if now_cfg_timezone < end_cfg_timezone - datetime.timedelta(minutes=interval):
        # Get future weather starting at either now time or start time. Start time only if it is in the future.
        future_weather = weather_fetch(start=max(now_cfg_timezone, start_cfg_timezone), end=end_cfg_timezone, interval=interval)
        outside_temperatures[max(now_cfg_timezone, start_cfg_timezone):end_cfg_timezone] = future_weather.values

    # Combining historic data with outside_temperatures correctly if exists.
    if now_cfg_timezone > start_cfg_timezone + datetime.timedelta(minutes=interval):
        historic_weather = get_historic_outside_data(start_utc, min(end_utc, utc_now))

        # Convert historic_weather to cfg timezone.
        historic_weather.index = historic_weather.index.tz_convert(tz=cfg_building["Pytz_Timezone"])

        # Make sure historic weather has correct interval and start to end times.
        historic_weather = utils.smart_resample(historic_weather,
                                                          start_cfg_timezone, min(end_cfg_timezone, now_cfg_timezone),
                                                          interval)

        # Populate outside data
        outside_temperatures[start_cfg_timezone:min(end_cfg_timezone, now_cfg_timezone)] = historic_weather.values

    return outside_temperatures

if __name__ == "__main__":
    building = "ciee"
    cfg_building = utils.get_config(building)
    pytz_tz = pytz.timezone(cfg_building["Pytz_Timezone"])
    end = utils.get_utc_now().astimezone(tz=pytz_tz)
    start = end - datetime.timedelta(hours=5)


    print("start", start)
    print("end", end)
    print(get_historic_outside_data(building, start, end, 15))