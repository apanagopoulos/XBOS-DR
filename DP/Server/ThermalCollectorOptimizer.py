import pandas as pd
import utils
import itertools
import datetime
import pytz
import os

def create_schedule(start, end, building, zones, interval=15):
    """

    :param start: 
    :param end: 
    :param building: 
    :param zones: 
    :param interval: 
    :return: 
    """
    time_range = pd.date_range(start, end, freq=str(interval) + 'T')
    schedule = pd.DataFrame(data=utils.NO_ACTION, columns=zones, index=time_range)

    # cartesian product of all zones and then for each zone cartesian product of heat and cool.
    actions = [utils.HEATING_ACTION, utils.COOLING_ACTION]

    curr_timestep = 0
    # we set up 15 min for every type of action for each zone.
    for iter_zone in zones:
        for a in actions:
            if "CS" in iter_zone and a == 1:
                continue
            schedule.iloc[curr_timestep][iter_zone] = a
            curr_timestep += 1

    for zone1, zone2 in itertools.combinations(zones, r=2):
        for a1, a2 in itertools.product(actions, actions):
            if start + datetime.timedelta(minutes=interval)*curr_timestep >= end:
                break
            schedule.iloc[curr_timestep][zone1] = a1
            schedule.iloc[curr_timestep][zone2] = a2
            curr_timestep += 1

    return schedule


class Schedule_Optimizer:

    def __init__(self, building, client, schedule, debug=None):
        self.schedule = schedule

    # in case that the mpc doesnt work properly run this
    def advise(self, start, end, zones, cfg_building, cfg_zones, thermal_model=None,
               zone_temperatures=None, consumption_storage=None):
        """

        :param start: 
        :param end: 
        :param zones: 
        :param cfg_building: 
        :param cfg_zones: 
        :param thermal_model: 
        :param zone_temperatures: 
        :param consumption_storage: 
        :return: 
        """
        pytz_timezone = pytz.timezone(cfg_building["Pytz_Timezone"])
        start_cfg_timezone = start.astimezone(tz=pytz_timezone)
        end_cfg_timezone = end.astimezone(tz=pytz_timezone)

        self.schedule.index = self.schedule.index.tz_convert(cfg_building["Pytz_Timezone"])



        return self.schedule.iloc[self.schedule.index.get_loc(start_cfg_timezone, method='nearest')].to_dict(), None





if __name__ == "__main__":
    start = utils.get_utc_now()
    end = start + datetime.timedelta(hours = 10)
    building = "ciee"
    zones = utils.get_zones(building)
    cfg_building = utils.get_config(building)
    cfg_zones = {iter_zone: utils.get_zone_config(building, iter_zone) for iter_zone in zones}
    so = Schedule_Optimizer(start, end, create_schedule(start, end, building, zones))
    print(so.advise(start, end, zones, cfg_building, cfg_zones))