import datetime
import sys

import pytz

import utils
from DataManager import DataManager


# TODO DR EVENT needs fixing

class NormalSchedule:
    # building, client, num_threads=1, debug=False
    def __init__(self, building, client, num_threads=1, debug=None):
        """
        The normal schedule class that will be share by the zones in a thread. 
        :param building: 
        :param client: 
        :param num_threads: 
        :param debug: 
        """
        self.building = building
        self.debug = debug
        self.client = client

        # set threads
        self.num_threads = num_threads


    # in case that the mpc doesnt work properly run this
    def advise(self, start, end, zones, cfg_building, cfg_zones, thermal_model,
               zone_temperatures, consumption_storage):
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

        # Get all datamanagers.
        datamanagers_zones = {
        iter_zone: DataManager(cfg_building, cfg_zones[iter_zone], None, now=start, zone=iter_zone) for iter_zone in
        zones}

        optimals_setpoints = {}

        # find the setpoints for each zone.
        for iter_zone in zones:
            cfg_zone = cfg_zones[iter_zone]
            # TODO , fix to get baseline.
            baseline_schedules = utils.get_comfortband_matrix(self.building, zones, start, end, 15, datamanagers_zones)

            setpoint_start = baseline_schedules.loc[start_cfg_timezone]
            setpoint_high = setpoint_start["t_high"]
            setpoint_low = setpoint_start["t_low"]

            if utils.is_DR(start, cfg_building):
                setpoint_high += cfg_zone["Advise"]["Baseline_Dr_Extend_Percent"]
                setpoint_low -= cfg_zone["Advise"]["Baseline_Dr_Extend_Percent"]

            optimals_setpoints[iter_zone] = {"heating_setpoint": setpoint_low, "cooling_setpoint": setpoint_high}

        return None, optimals_setpoints


if __name__ == '__main__':

    import yaml

    with open("config_file.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    from xbos import get_client
    from xbos.services.hod import HodClient

    if cfg["Server"]:
        client = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        client = get_client()

    hc = HodClient("xbos/hod", client)

    q = """SELECT ?uri ?zone FROM %s WHERE {
			?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
			?tstat bf:uri ?uri .
			?tstat bf:controls/bf:feeds ?zone .
			};""" % cfg["Building"]

    from xbos.devices.thermostat import Thermostat

    for tstat in hc.do_query(q)['Rows']:
        print tstat
        with open("Buildings/" + cfg["Building"] + "/ZoneConfigs/" + tstat["?zone"] + ".yml", 'r') as ymlfile:
            advise_cfg = yaml.load(ymlfile)
        NS = NormalSchedule(cfg, Thermostat(client, tstat["?uri"]), advise_cfg)
        NS.normal_schedule()
