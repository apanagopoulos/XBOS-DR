import datetime
import sys
import threading
import time
import traceback

import pandas as pd
import pytz
import yaml
import math

import utils
from DataManager import DataManager
from ThermalDataManager import ThermalDataManager
from NormalSchedule import NormalSchedule

sys.path.insert(0, './MPC')
from Advise import Advise
from ThermalModel import *
# from AverageThermalModel import *

sys.path.insert(0, '../Utils')
import Debugger

from xbos import get_client
from xbos.services.hod import HodClient



# TODO set up a moving average for how long it took for action to take place.
# the main controller
def hvac_control(cfg, advise_cfg, tstats, client, thermal_model, zone, building):
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
        prices = dataManager.prices()
        building_setpoints = dataManager.building_setpoints()

        # need to set weather predictions for every loop and set current zone temperatures and fit the model given the new data (if possible).
        # NOTE: call setZoneTemperaturesAndFit before setWeahterPredictions
        # TODO Double Check if update to new thermal model was correct
        thermal_model.set_temperatures_and_fit(zone_temperatures, interval=cfg["Interval_Length"],
                                               now=now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])))

        # TODO Look at the weather_fetch function and make sure correct locks are implemented and we are getting the right data.
        weather = dataManager.weather_fetch()
        thermal_model.set_weather_predictions(weather)

        if (cfg["Pricing"]["DR"] and utils.in_between(now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).time(),
                                                      utils.get_time_datetime(cfg["Pricing"]["DR_Start"]),
                                                      utils.get_time_datetime(cfg["Pricing"]["DR_Finish"]))):  # \
            # or now.weekday() == 4:  # TODO REMOVE ALLWAYS HAVING DR ON FRIDAY WHEN DR SUBSCRIBE IS IMPLEMENTED
            DR = True
        else:
            DR = False

        adv_start = time.time()
        adv = Advise([zone],  # array because we might use more than one zone. Multiclass approach.
                     now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])),
                     dataManager.preprocess_occ(),
                     [tstat_temperature],
                     thermal_model,
                     prices,
                     advise_cfg["Advise"]["General_Lambda"],
                     advise_cfg["Advise"]["DR_Lambda"],
                     DR,
                     cfg["Interval_Length"],
                     advise_cfg["Advise"]["MPCPredictiveHorizon"],
                     advise_cfg["Advise"]["Heating_Consumption"],
                     advise_cfg["Advise"]["Cooling_Consumption"],
                     advise_cfg["Advise"]["Ventilation_Consumption"],
                     advise_cfg["Advise"]["Thermal_Precision"],
                     advise_cfg["Advise"]["Occupancy_Obs_Len_Addition"],
                     building_setpoints,
                     advise_cfg["Advise"]["Occupancy_Sensors"],
                     safety_constraints)

        action = adv.advise()
        adv_end = time.time()

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

    # Plot the MPC graph.
    if advise_cfg["Advise"]["Print_Graph"]:
        adv.g_plot(zone)

    # Log the information related to the current MPC
    Debugger.debug_print(now, building, zone, adv, safety_constraints, prices, building_setpoints, adv_end - adv_start, file=True)

    # try to commit the changes to the thermostat, if it doesnt work 10 times in a row ignore and try again later
    for i in range(advise_cfg["Advise"]["Thermostat_Write_Tries"]):
        try:
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



class ZoneThread(threading.Thread):
    def __init__(self, cfg_filename, tstats, zone, client, thermal_model, building):
        threading.Thread.__init__(self)
        self.cfg_filename = cfg_filename
        self.tstats = tstats
        self.zone = zone
        self.client = client
        self.building = building
        self.thermal_model = thermal_model

    def run(self):
        starttime = time.time()
        action_data = None
        while True:
            try:
                # Reloading the config everytime we iterate.
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
                                                          self.zone, self.building)
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
            print("This process is for building %s" % cfg["Building"]) # TODO Rethink. now every thread will write this.
            # Wait for the next interval.
            time.sleep(60. * float(cfg["Interval_Length"]) - (
            (time.time() - starttime) % (60. * float(cfg["Interval_Length"]))))

            # end program if setpoints have been changed. (If not writing to tstat we don't want this)
            if action_data is not None and utils.has_setpoint_changed(self.tstats[self.zone], action_data, self.zone, self.building):
                print("Ending program for zone %s due to manual setpoint changes. \n" % self.zone)
                return


if __name__ == '__main__':
    # TODO check for comfortband height and whether correctly implemented
    building = sys.argv[1]

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

    hc = HodClient("xbos/hod", client)

    tstats = utils.get_thermostats(client, hc, cfg["Building"])
    threads = []

    # --- Thermal Model Init ------------
    # initialize and fit thermal model

    # only single stage cooling buildings get to retrive data. otherwise takes too long.
    if building in ["north-berkeley-senior-center", "ciee", "avenal-veterans-hall", "hayward-station-1", "hayward-station-8", "orinda-community-center"]:
        thermal_data = utils.get_data(cfg=cfg, client=client, days_back=150, force_reload=False)

        zone_thermal_models = {}
        for zone, zone_data in thermal_data.items():
            # Concat zone data to put all data together and filter such that all datapoints have dt != 1
            filtered_zone_data = zone_data[zone_data["dt"] == 5]
            if zone != "HVAC_Zone_Please_Delete_Me":
                zone_thermal_models[zone] = MPCThermalModel(zone=zone, thermal_data=filtered_zone_data,
                                                        interval_length=15, thermal_precision=0.05)
    else:
        zone_thermal_models = {}
        for zone in tstats.keys():
            zone_thermal_models[zone] = None


    print("Trained Thermal Model")
    # --------------------------------------



    for zone, tstat in tstats.items():
        # TODO only because we want to only run on the basketball courts.
        if building != "jesse-turner-center" or "Basketball" in zone:
            thread = ZoneThread(yaml_filename, tstats, zone, client, zone_thermal_models[zone], cfg["Building"])
            thread.start()
            threads.append(thread)

    for t in threads:
        t.join()
