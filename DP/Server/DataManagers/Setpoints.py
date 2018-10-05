import pandas as pd
import numpy as np
import datetime
import pytz

import sys
sys.path.append("../")
import utils



def get_day_comfortband(building, zone, date, interval):
    """
    Gets the comfortband from the zone configuration file. Uses the provided date for which the comfortband should
    hold. If there are nan's in the comfortband, then those will be replaced with the saftysetpoints.
    Will only look at the starting times of the setpoints in config file. Assumes everything between starts has
    same setpoint.
    :param date: The date for which we want to get the comfortband from config. Timezone aware.
    :param freq: (int) The frequency in minutes of time series. Default is one minute.
    :return: pd.df columns=t_high, t_low with time_series index for the date provided and in timezone aware
     datetime with timezone as provided by the configuration file.
    """
    cfg_building = utils.get_config(building)
    cfg_zone = utils.get_zone_config(building, zone)
    pytz_tz = pytz.timezone(cfg_building["Pytz_Timezone"])


    # Set the date to the controller timezone.
    date = date.astimezone(tz=pytz_tz)

    setpoints_array = cfg_zone["Advise"]["Comfortband"]
    df_safety = get_day_safety(building, zone, date, interval)

    weekday = date.weekday()

    date_setpoints = np.array(setpoints_array[weekday])

    start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=1)
    date_range = pd.date_range(start=start_date, end=end_date, freq=str(interval)+"T")

    # create nan filled dataframe and populate it
    df_setpoints = pd.DataFrame(columns=["t_high", "t_low"], index=date_range, dtype=float)


    # sanity check that we don't have any timeperiods where no setpoints are given
    last_end = None

    for interval_setpoints in date_setpoints:
        start, end, t_low, t_high = interval_setpoints
        start = utils.combine_date_time(start, date)

        # sanity check that we don't have any timeperiods where no setpoints are given
        if last_end is not None and start != last_end:
            raise Exception("Data was given where for some time period no setpoints were given.")
        last_end = utils.combine_date_time(end, date)

        # if we are going into the next day.
        if last_end <= start:
            last_end += datetime.timedelta(days=1)

        if t_low is None or t_high is None or t_low == "None" or t_high == "None":
            interval_safety = df_safety.loc[start]
            t_low, t_high = interval_safety["t_low"], interval_safety["t_high"]

        t_low = float(t_low)
        t_high = float(t_high)
        df_setpoints.loc[start, "t_high"] = t_high
        df_setpoints.loc[last_end, "t_high"] = t_high
        df_setpoints.loc[start, "t_low"] = t_low
        df_setpoints.loc[last_end, "t_low"] = t_low

    df_setpoints = utils.smart_resample(df_setpoints.sort_index(), start_date, end_date, interval)

    return df_setpoints


def get_day_safety(building, zone, date, interval=1):
    """
    Gets the safety_setpoints from the zone configuration file. Uses the provided date for which the setpoints should
    hold. Cannot have Nan values.
    :param date: The date for which we want to get the safety setpoints from config. Timezone aware.
    :param freq: The frequency of time series. Default is one minute.
    :return: pd.df columns=t_high, t_low with time_series index for the date provided and in timezone aware
     datetime as provided by the configuration file.
    """
    # Set the date to the controller timezone.
    cfg_building = utils.get_config(building)
    cfg_zone = utils.get_zone_config(building, zone)
    pytz_tz = pytz.timezone(cfg_building["Pytz_Timezone"])

    date = date.astimezone(tz=pytz_tz)

    setpoints_array = cfg_zone["Advise"]["SafetySetpoints"]

    weekday = date.weekday()

    date_setpoints = np.array(setpoints_array[weekday])

    start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=1)
    date_range = pd.date_range(start=start_date, end=end_date, freq=str(interval)+"T")

    # create nan filled dataframe and populate it
    df_setpoints = pd.DataFrame(columns=["t_high", "t_low"], index=date_range)

    # helper variable to check for discontinuities in data.
    last_end = None

    for interval_setpoints in date_setpoints:
        start, end, t_low, t_high = interval_setpoints
        start = utils.combine_date_time(start, date)

        # sanity check that we don't have any timeperiods where no setpoints are given
        if last_end is not None and start != last_end:
            raise Exception("Data was given where for some time period no setpoints were given.")
        last_end = utils.combine_date_time(end, date)

        # if we are going into the next day.
        if last_end <= start:
            last_end += datetime.timedelta(days=1)

        if t_low is None or t_high is None or t_low == "None" or t_high == "None":
            raise Exception("Safety should have no None.")

        t_low = float(t_low)
        t_high = float(t_high)
        df_setpoints.loc[start, "t_high"] = t_high
        df_setpoints.loc[last_end, "t_high"] = t_high
        df_setpoints.loc[start, "t_low"] = t_low
        df_setpoints.loc[last_end, "t_low"] = t_low

    df_setpoints = utils.smart_resample(df_setpoints.sort_index(), start_date, end_date, interval)

    return df_setpoints


def get_setpoints(building, zone, start, end, interval, method):
    """
    Gets the comfortband of a zone from start to end in interval minutes frequency
    :param building: string
    :param zone: string
    :param start: datetime. timezone aware
    :param end: datetime. timezone aware.
    :param interval: float. minutes
    :param method: string ["comfortband", "safety"] decides which setpoints to get.
    :return:
    """
    assert start <= end

    curr_time = start
    all_setpoints = []

    # get setpoints for each day
    while curr_time.date() <= end.date():
        if method == "comfortband":
            day_setpoints = get_day_comfortband(building, zone, curr_time, interval)
        elif method == "safety":
            day_setpoints = get_day_safety(building, zone, curr_time, interval)
        else:
            raise Exception("Invalid method given for setpoints.")
        all_setpoints.append(day_setpoints)
        curr_time += datetime.timedelta(days=1)

    setpoints_df = pd.concat(all_setpoints)

    # fixes the dataset to be of right interval and from start to end.
    setpoints_df = utils.smart_resample(setpoints_df.sort_index(), start, end, interval)

    # set timezone to local building timezone
    cfg_building = utils.get_config(building)
    setpoints_df = setpoints_df.tz_convert(cfg_building["Pytz_Timezone"])

    return setpoints_df


def get_comfortband(building, zone, start, end, interval):
    return get_setpoints(building, zone, start, end, interval, method="comfortband")


def get_safety(building, zone, start, end, interval):
    return get_setpoints(building, zone, start, end, interval, method="safety")




if __name__ == "__main__":
    building = "ciee"
    zones = utils.get_zones(building)
    zone = zones[0]

    print("zone", zone)
    end = utils.get_utc_now()
    start = end - datetime.timedelta(days=2)
    print("start", start)
    print("end", end)
    # print(get_day_comfortband(building, zone, start, 15))
    print(get_comfortband(building, zone, start, start + datetime.timedelta(minutes=4001), 15))

