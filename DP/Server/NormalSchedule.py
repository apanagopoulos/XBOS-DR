import datetime
import sys

import pytz

from DataManager import DataManager


# TODO DR EVENT needs fixing

class NormalSchedule:
    def __init__(self, cfg, t_stat, advise_cfg, now=None):
        """
        
        :param cfg: 
        :param t_stat: 
        :param advise_cfg: 
        :param now: in UTC time. If None (so no now is passed), take the current time. 
        
        """
        self.cfg = cfg
        self.advise_cfg = advise_cfg
        if now is None:
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))
        self.now = now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"]))
        print self.now
        self.tstat = t_stat
        self.zone = advise_cfg["Zone"]

    # in case that the mpc doesnt work properly run this
    def normal_schedule(self, debug=False):
        """
        
        :param debug: Whether to actuate tstat.
        :return: 
        """
        def in_between(now, start, end):
            if start < end:
                return start <= now < end
            elif end < start:
                return start <= now or now < end
            else:
                return True

        def getDatetime(date_string):
            """Gets datetime from string with format HH:MM. Should be changed to datetime in-built function. """
            return datetime.time(int(date_string.split(":")[0]), int(date_string.split(":")[1]))

        setpoints_array = self.advise_cfg["Advise"]["Baseline"][self.now.weekday()]

        for j in setpoints_array:
            if in_between(self.now.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
                          datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
                SetpointLow = j[2]
                SetpointHigh = j[3]
                break

        dataManager = DataManager(self.cfg, self.advise_cfg, None, now=self.now, zone=self.zone)
        Safety_temps = dataManager.safety_constraints()

        if not isinstance(SetpointLow, (int, float, long)):
            SetpointLow = Safety_temps[0][0]
        if not isinstance(SetpointHigh, (int, float, long)):
            SetpointHigh = Safety_temps[0][1]

        if (self.cfg["Pricing"]["DR"] and in_between(self.now.time(), getDatetime(self.cfg["Pricing"]["DR_Start"]),
                                                     getDatetime(self.cfg["Pricing"]["DR_Finish"]))) \
                or self.now.weekday() == 4:  # TODO REMOVE ALLWAYS HAVING DR ON FRIDAY WHEN DR SUBSCRIBE IS IMPLEMENTED
            SetpointHigh += self.advise_cfg["Advise"]["Baseline_Dr_Extend_Percent"]
            SetpointLow -= self.advise_cfg["Advise"]["Baseline_Dr_Extend_Percent"]

        # Making sure that the different between setpointHigh and Low is at least the Comfortband
        if SetpointHigh - SetpointLow < self.advise_cfg["Advise"]["Minimum_Comfortband_Height"]:
            raise Exception(
                "Warning, the difference between SetpointHigh and SetpointLow is too narrow. Difference: %s. Check the config file schedule." % str(
                    SetpointHigh - SetpointLow))

        # making sure that we are not exceeding the Safety temps.
        # Only violates the Comfortband height if safefy temperatures violate it.
        if SetpointLow < Safety_temps[0][0]:
            diff = Safety_temps[0][0] - SetpointLow
            SetpointLow = Safety_temps[0][0]
            SetpointHigh = min(Safety_temps[0][1], SetpointHigh + diff)

        elif SetpointHigh > Safety_temps[0][1]:
            diff = SetpointHigh - Safety_temps[0][1]
            SetpointHigh = Safety_temps[0][1]
            SetpointLow = max(Safety_temps[0][0], SetpointLow - diff)

        p = {"override": True, "heating_setpoint": SetpointLow, "cooling_setpoint": SetpointHigh, "mode": 3}

        for i in range(self.advise_cfg["Advise"]["Thermostat_Write_Tries"]):
            try:
                if not debug:
                    self.tstat.write(p)
                print("For zone: %s writing Baseline: %s" % (self.zone, str(p)))
                break
            except:
                if i == self.advise_cfg["Advise"]["Thermostat_Write_Tries"] - 1:
                    e = sys.exc_info()[0]
                    print e
                    return False, None
                continue
        return True, p

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
