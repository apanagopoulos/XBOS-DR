import datetime

import numpy as np

import pandas as pd
import pytz
from xbos import get_client
from xbos.services import mdal
from xbos.services.hod import HodClient

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.dirname(FILE_PATH) # The Server is Parent for now.

import sys
sys.path.append("./..")
import utils


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
        , 'Time': {'T0': start.astimezone(tz=pytz.utc).strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                   'T1': end.astimezone(tz=pytz.utc).strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                   'WindowSize': str(15) + 'min', # we are getting 15 min intervals because then we get fewer nan values. MDAL specific issue.
                   'Aligned': True}}

    mdal_outside_data = utils.get_mdal_data(mdal_client, mdal_query)
    return mdal_outside_data.tz_convert(utils.get_config(building)["Pytz_Timezone"])


def get_historic_outside_data(building, start, end, interval):
    # TODO what to do if all values are nan?
    mdal_data = _get_mdal_outside_data(building, start, end)
    preprocessed_data = _preprocess_mdal_outside_data(mdal_data)
    return utils.smart_resample(preprocessed_data, start, end, interval, method="interpolate")



if __name__ == "__main__":
    building = "ciee"
    cfg_building = utils.get_config(building)
    pytz_tz = pytz.timezone(cfg_building["Pytz_Timezone"])
    end =datetime.datetime.strptime("17/09/2018 7:33:00", "%d/%m/%Y %H:%M:%S").replace(tzinfo=pytz.utc)
    start = datetime.datetime.strptime("17/09/2018 6:30:00", "%d/%m/%Y %H:%M:%S").replace(tzinfo=pytz.utc)


    print("start", start)
    print("end", end)
    print(get_historic_outside_data(building, start, end, 15))