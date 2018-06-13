import datetime
import sys
import threading
import time
import traceback
import math

import pytz
import yaml

from ControllerDataManager import ControllerDataManager
from DataManager import DataManager
from NormalSchedule import NormalSchedule

sys.path.insert(0, 'MPC')
from Advise import Advise
from ThermalModel import *

from xbos import get_client
from xbos.services.hod import HodClient
from xbos.devices.thermostat import Thermostat


def in_between(now, start, end):
    if start < end:
        return start <= now < end
    elif end < start:
        return start <= now or now < end
    else:
        return True


def getDatetime(date_string):
    """Gets datetime from string with format HH:MM.
    :param date_string: string of format HH:MM
    :returns datetime.time() object with no associated timzone. """
    return datetime.datetime.strptime(date_string, "%H:%M").time()


# TODO set up a moving average for how long it took for action to take place.
# the main controller
def hvac_control(cfg, advise_cfg, tstats, client, thermal_model, zone):
    """
    
    :param cfg: 
    :param advise_cfg: 
    :param tstats: 
    :param client: 
    :param thermal_model: 
    :param zone: 
    :return: boolean, dict. Success Boolean indicates whether writing action has succeeded. Dictionary {cooling_setpoint: float,
    heating_setpoint: float, override: bool, mode: int} and None if success boolean is flase. 
    """

    # now in UTC time.
    now = pytz.timezone("UTC").localize(datetime.datetime.utcnow())
    try:
        zone_temperatures = {dict_zone: dict_tstat.temperature for dict_zone, dict_tstat in tstats.items()}
        tstat = tstats[zone]
        tstat_temperature = zone_temperatures[zone]  # to make sure we get all temperatures at the same time

        dataManager = DataManager(cfg, advise_cfg, client, zone, now=now)
        safety_constraints = dataManager.safety_constraints()

        # need to set weather predictions for every loop and set current zone temperatures and fit the model given the new data (if possible).
        # NOTE: call setZoneTemperaturesAndFit before setWeahterPredictions
        # TODO Double Check if update to new thermal model was correct
        thermal_model.set_temperatures_and_fit(zone_temperatures, interval=cfg["Interval_Length"],
                                               now=now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])))

        # TODO Look at the weather_fetch function and make sure correct locks are implemented and we are getting the right data.
        weather = dataManager.weather_fetch()
        thermal_model.set_weather_predictions(weather)

        if (cfg["Pricing"]["DR"] and in_between(now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).time(),
                                                getDatetime(cfg["Pricing"]["DR_Start"]),
                                                getDatetime(cfg["Pricing"]["DR_Finish"]))): #\
                #or now.weekday() == 4:  # TODO REMOVE ALLWAYS HAVING DR ON FRIDAY WHEN DR SUBSCRIBE IS IMPLEMENTED
            DR = True
        else:
            DR = False

        adv = Advise([zone],  # array because we might use more than one zone. Multiclass approach.
                     now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])),
                     dataManager.preprocess_occ(),
                     [tstat_temperature],
                     thermal_model,
                     dataManager.prices(),
                     advise_cfg["Advise"]["General_Lambda"],
                     advise_cfg["Advise"]["DR_Lambda"],
                     DR,
                     cfg["Interval_Length"],
                     advise_cfg["Advise"]["MPCPredictiveHorizon"],
                     advise_cfg["Advise"]["Print_Graph"],
                     advise_cfg["Advise"]["Heating_Consumption"],
                     advise_cfg["Advise"]["Cooling_Consumption"],
                     advise_cfg["Advise"]["Ventilation_Consumption"],
                     advise_cfg["Advise"]["Thermal_Precision"],
                     advise_cfg["Advise"]["Occupancy_Obs_Len_Addition"],
                     dataManager.building_setpoints(),
                     advise_cfg["Advise"]["Occupancy_Sensors"],
                     safety_constraints)

        action = adv.advise()


    except Exception:

        print(traceback.format_exc())
        # TODO Find a better way for exceptions
        return False

    # action "0" is Do Nothing, action "1" is Heating, action "2" is Cooling
    if action == "0":
        heating_setpoint = tstat_temperature - advise_cfg["Advise"]["Minimum_Comfortband_Height"] / 2.
        cooling_setpoint = tstat_temperature + advise_cfg["Advise"]["Minimum_Comfortband_Height"] / 2.

        if heating_setpoint < safety_constraints[0][0]:
            heating_setpoint = safety_constraints[0][0]

            if (cooling_setpoint - heating_setpoint) < advise_cfg["Advise"]["Minimum_Comfortband_Height"]:
                cooling_setpoint = min(safety_constraints[0][1],
                                       heating_setpoint + advise_cfg["Advise"]["Minimum_Comfortband_Height"])

        elif cooling_setpoint > safety_constraints[0][1]:
            cooling_setpoint = safety_constraints[0][1]

            if (cooling_setpoint - heating_setpoint) < advise_cfg["Advise"]["Minimum_Comfortband_Height"]:
                heating_setpoint = max(safety_constraints[0][0],
                                       cooling_setpoint - advise_cfg["Advise"]["Minimum_Comfortband_Height"])

        # round to integers since the thermostats round internally.
        heating_setpoint = math.floor(heating_setpoint)
        cooling_setpoint = math.ceil(cooling_setpoint)

        p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint, "mode": 3}
        print "Doing nothing"

    # TODO Rethink how we set setpoints for heating and cooling and for DR events.
    # heating
    elif action == "1":
        heating_setpoint = tstat_temperature + 2 * advise_cfg["Advise"]["Hysterisis"]
        cooling_setpoint = heating_setpoint + advise_cfg["Advise"]["Minimum_Comfortband_Height"]

        if cooling_setpoint > safety_constraints[0][1]:
            cooling_setpoint = safety_constraints[0][1]

            # making sure we are in the comfortband
            if (cooling_setpoint - heating_setpoint) < advise_cfg["Advise"]["Minimum_Comfortband_Height"]:
                heating_setpoint = max(safety_constraints[0][0],
                                       cooling_setpoint - advise_cfg["Advise"]["Minimum_Comfortband_Height"])

        # round to integers since the thermostats round internally.
        heating_setpoint = math.ceil(heating_setpoint)
        cooling_setpoint = math.ceil(cooling_setpoint)

        p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint, "mode": 3}
        print "Heating"

    # cooling
    elif action == "2":
        cooling_setpoint = tstat_temperature - 2 * advise_cfg["Advise"]["Hysterisis"]
        heating_setpoint = cooling_setpoint - advise_cfg["Advise"]["Minimum_Comfortband_Height"]

        if heating_setpoint < safety_constraints[0][0]:
            heating_setpoint = safety_constraints[0][0]

            # making sure we are in the comfortband
            if (cooling_setpoint - heating_setpoint) < advise_cfg["Advise"]["Minimum_Comfortband_Height"]:
                cooling_setpoint = min(safety_constraints[0][1],
                                       heating_setpoint + advise_cfg["Advise"]["Minimum_Comfortband_Height"])

        # round to integers since the thermostats round internally.
        heating_setpoint = math.floor(heating_setpoint)
        cooling_setpoint = math.floor(cooling_setpoint)

        p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint, "mode": 3}
        print "Cooling"
    else:
        print "Problem with action."
        return False, None

    print("Zone: " + zone + ", action: " + str(p))

    # try to commit the changes to the thermostat, if it doesnt work 10 times in a row ignore and try again later
    for i in range(advise_cfg["Advise"]["Thermostat_Write_Tries"]):
        try:
            # TODO Uncomment
            tstat.write(p)
            thermal_model.set_last_action(
                action)  # TODO Document that this needs to be set after we are sure that writing has succeeded.
            break
        except:
            if i == advise_cfg["Advise"]["Thermostat_Write_Tries"] - 1:
                e = sys.exc_info()[0]
                print e
                return False, None
            continue

    return True, p


def has_setpoint_changed(tstat, setpoint_data, zone):
    """
    Checks if thermostats was manually changed and prints warning. 
    :param tstat: Tstat object we want to look at. 
    :param setpoint_data: dict which has keys {"heating_setpoint": bool, "cooling_setpoint": bool} and corresponds to
            the setpoint written to the thermostat by MPC. 
    :param zone: Name of the zone to print correct messages. 
    :return: Bool. Whether tstat setpoints are equal to setpoints written to tstat.
    """
    WARNING_MSG = "WARNING. %s has been manually changed in zone %s. Setpoint is at %s from expected %s. " \
                  "Setting override to False and intiatiating program stop."
    flag_changed = False
    if tstat.cooling_setpoint != setpoint_data["cooling_setpoint"]:
        flag_changed = True
        print(WARNING_MSG % ("cooling setpoint", zone, tstat.cooling_setpoint, setpoint_data["cooling_setpoint"]))
    if tstat.heating_setpoint != setpoint_data["heating_setpoint"]:
        flag_changed = True
        print(WARNING_MSG % ("heating setpoint", zone, tstat.heating_setpoint, setpoint_data["heating_setpoint"]))

    # write override false so the local schedules can take over again.
    if flag_changed:
        tstat.write({"override": False})
    return flag_changed


class ZoneThread(threading.Thread):
    def __init__(self, cfg_filename, tstats, zone, client, thermal_model):
        threading.Thread.__init__(self)
        self.cfg_filename = cfg_filename
        self.tstats = tstats
        self.zone = zone
        self.client = client

        self.thermal_model = thermal_model

    def run(self):
        starttime = time.time()
        action_data = None
        while True:
            try:
                with open(self.cfg_filename, 'r') as ymlfile:
                    cfg = yaml.load(ymlfile)
                with open("Buildings/" + cfg["Building"] + "/ZoneConfigs/" + self.zone + ".yml", 'r') as ymlfile:
                    advise_cfg = yaml.load(ymlfile)
            except:
                print "There is no " + self.zone + ".yml file under ZoneConfigs folder."
                return  # TODO MAKE THIS RUN NORMAL SCHEDULE SOMEHOW WHEN NO ZONE CONFIG EXISTS

            normal_schedule_succeeded = None  # initialize

            if advise_cfg["Advise"]["MPC"]:
                # Run MPC. Try up to advise_cfg["Advise"]["Thermostat_Write_Tries"] to find and write action.
                count = 0
                succeeded = False
                while not succeeded:
                    succeeded, action_data = hvac_control(cfg, advise_cfg, self.tstats, self.client, self.thermal_model,
                                                          self.zone)
                    if not succeeded:
                        time.sleep(10)
                        if count == advise_cfg["Advise"]["Thermostat_Write_Tries"]:
                            print("Problem with MPC, entering normal schedule.")
                            normal_schedule = NormalSchedule(cfg, tstat, advise_cfg)
                            normal_schedule_succeeded, action_data = normal_schedule.normal_schedule()
                            break
                        count += 1
            else:
                # go into normal schedule
                normal_schedule = NormalSchedule(cfg, self.tstats[self.zone], advise_cfg)
                normal_schedule_succeeded, action_data = normal_schedule.normal_schedule()

            # TODO if normal schedule fails then real problems
            if normal_schedule_succeeded is not None and not normal_schedule_succeeded:
                print("WARNING, normal schedule has not succeeded.")

            print datetime.datetime.now()
            # Wait for the next interval.
            time.sleep(60. * float(cfg["Interval_Length"]) - (
            (time.time() - starttime) % (60. * float(cfg["Interval_Length"]))))

            # end program if setpoints have been changed. (If not writing to tstat we don't want this)
            if action_data is not None and has_setpoint_changed(self.tstats[self.zone], action_data, self.zone):
                print("Ending program for zone %s due to manual setpoint changes. \n" % self.zone)
                return


if __name__ == '__main__':
    # TODO check for comfortband height and whether correctly implemented

    # read from config file
    try:
        yaml_filename = "Buildings/%s/%s.yml" % (sys.argv[1], sys.argv[1])
    except:
        sys.exit("Please specify the configuration file as: python2 controller.py config_file.yaml")

    with open(yaml_filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    if cfg["Server"]:
        client = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        client = get_client()

    # --- Thermal Model Init ------------
    # initialize and fit thermal model
    import pickle

    try:
        with open("Thermal Data/demo_" + cfg["Building"], "r") as f:
            thermal_data = pickle.load(f)

    except:
        controller_dataManager = ControllerDataManager(cfg, client)
        thermal_data = controller_dataManager.thermal_data(days_back=20)
        with open("Thermal Data/demo_" + cfg["Building"], "wb") as f:
            pickle.dump(thermal_data, f)

    # TODO INTERVAL SHOULD NOT BE IN config_file.yml, THERE SHOULD BE A DIFFERENT INTERVAL FOR EACH ZONE
    zone_thermal_models = {zone: MPCThermalModel(zone, zone_thermal_data, interval_length=cfg["Interval_Length"],
                                                 thermal_precision=cfg["Thermal_Precision"])
                           for zone, zone_thermal_data in thermal_data.items()}
    print("Trained Thermal Model")
    # --------------------------------------


    with open(yaml_filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    hc = HodClient("xbos/hod", client)

    q = """SELECT ?uri ?zone FROM %s WHERE {
        ?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
        ?tstat bf:uri ?uri .
        ?tstat bf:controls/bf:feeds ?zone .
        };""" % cfg["Building"]

    # Start of FIX for missing Brick query
    thermostat_query = """SELECT ?zone ?uri FROM  %s WHERE {
              ?tstat rdf:type brick:Thermostat .
              ?tstat bf:controls ?RTU .
              ?RTU rdf:type brick:RTU .
              ?RTU bf:feeds ?zone. 
              ?zone rdf:type brick:HVAC_Zone .
              ?tstat bf:uri ?uri.
              };"""
    q = thermostat_query % cfg["Building"]
    # End of FIX - delete when Brick is fixed

    threads = []
    tstat_query_data = hc.do_query(q)['Rows']
    print(tstat_query_data)
    tstats = {tstat["?zone"]: Thermostat(client, tstat["?uri"]) for tstat in tstat_query_data}

    for zone, tstat in tstats.items():
        thread = ZoneThread(yaml_filename, tstats, zone, client, zone_thermal_models[zone])
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()
