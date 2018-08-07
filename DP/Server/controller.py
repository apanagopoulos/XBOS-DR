import datetime
import math
import sys
import threading
import time
import traceback

import pandas as pd
import pytz
import yaml

import utils
from DataManager import DataManager
from NormalSchedule import NormalSchedule
from SimulationTstat import SimulationTstat
from ThermalDataManager import ThermalDataManager

sys.path.insert(0, './MPC')
sys.path.insert(0, './MPC/ThermalModels')
from Advise import Advise
from MPCThermalModel import MPCThermalModel

# from AverageThermalModel import *

sys.path.insert(0, '../Utils')
sys.path.append("./Lights")
import Debugger
import lights
import MPCLogger

from xbos.services.hod import HodClient


# TODO set up a moving average for how long it took for action to take place.
# the main controller
def hvac_control(cfg, advise_cfg, tstats, client, thermal_model, zone, building, now,
                 debug=False, simulate=False):
    """
    
    :param cfg:
    :param advise_cfg:
    :param tstats:
    :param client:
    :param thermal_model:
    :param zone:
    :param now: datetime object in UTC which tells the control what now is.
    :param debug: wether to actuate the tstat.
    :param simulate: boolean whether to run the control as a simulation or to actually actuate.
    :return: boolean, dict. Success Boolean indicates whether writing action has succeeded. Dictionary {cooling_setpoint: float,
    heating_setpoint: float, override: bool, mode: int} and None if success boolean is flase.
    """

    try:

        zone_temperatures = {dict_zone: dict_tstat.temperature for dict_zone, dict_tstat in tstats.items()}
        tstat = tstats[zone]
        tstat_temperature = zone_temperatures[zone]  # to make sure we get all temperatures at the same time

        # get datamanagers
        dataManager = DataManager(cfg, advise_cfg, client, zone, now=now)
        thermal_data_manager = ThermalDataManager(cfg, client)

        safety_constraints = dataManager.safety_constraints()
        prices = dataManager.prices()
        building_setpoints = dataManager.building_setpoints()
        if simulate or not advise_cfg["Advise"]["Occupancy_Sensors"]:
            occ_predictions = dataManager.preprocess_occ_cfg()
        else:
            occ_predictions = dataManager.preprocess_occ_mdal()

        if not simulate:
            # TODO FIX THE UPDATE STEP. PUT THIS OUTSIDE OF HVAC CONTROL.
            # NOTE: call update before setWeatherPredictions and set_temperatures
            thermal_model.update(zone_temperatures, interval=cfg["Interval_Length"])

        # need to set weather predictions for every loop and set current zone temperatures.
        thermal_model.set_temperatures(zone_temperatures)

        # ===== Future and past outside temperature combine =====
        # Get correct weather predictions.
        # we might have that the given now is before the actual current time
        # hence need to get historic data and combine with weather predictions.

        # finding out where the historic/future intervals start and end.
        utc_now = utils.get_utc_now()

        # If simulation window is partially in the past and in the future
        if utils.in_between_datetime(utc_now,
                                     now, now + datetime.timedelta(hours=advise_cfg["Advise"]["MPCPredictiveHorizon"])):
            historic_start = now
            historic_end = utc_now
            future_start = utc_now
            future_end = now + datetime.timedelta(hours=advise_cfg["Advise"]["MPCPredictiveHorizon"])

        # If simulation window is fully in the future
        elif now >= utc_now:
            historic_start = None
            historic_end = None
            future_start = now
            future_end = now + datetime.timedelta(hours=advise_cfg["Advise"]["MPCPredictiveHorizon"])

        # If simulation window is fully in the past
        else:
            historic_start = now
            historic_end = now + datetime.timedelta(hours=advise_cfg["Advise"]["MPCPredictiveHorizon"])
            future_start = None
            future_end = None

        # Populating the outside_temperatures dictionary for MPC use. Ouput is in cfg timezone.
        outside_temperatures = {}
        if future_start is not None:
            # TODO implement end for weather_fetch
            future_weather = dataManager.weather_fetch(start=future_start)
            outside_temperatures = future_weather

        # Combining historic data with outside_temperatures correctly if exists.
        if historic_start is not None:
            historic_weather = thermal_data_manager._get_outside_data(historic_start, historic_end, inclusive=True)
            historic_weather = thermal_data_manager._preprocess_outside_data(historic_weather.values())

            # Down sample the historic weather to hourly entries, and take the mean for each hour.
            historic_weather = historic_weather.groupby([pd.Grouper(freq="1H")])["t_out"].mean()

            # Convert historic_weather to cfg timezone.
            historic_weather.index = historic_weather.index.tz_convert(tz=cfg["Pytz_Timezone"])

            # Popluate the outside_temperature array. If we have the simulation time in the past and future then
            # we will take a weighted averege of the historic and future temperatures in the hour in which
            # historic_end and future_start happen.
            for row in historic_weather.iteritems():
                row_time, t_out = row[0], row[1]

                # taking a weighted average of the past and future outside temperature since for now
                # we only have one outside temperature per hour.
                if row_time.hour in outside_temperatures and \
                                row_time.hour == historic_end.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).hour:

                    future_t_out = outside_temperatures[row_time.hour]

                    # Checking if start and end are in the same hour, because then we have to weigh the temperature by
                    # less.
                    if historic_end.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).hour ==\
                            historic_start.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).hour:
                        historic_weight = (historic_end - historic_start).seconds // 60
                    else:
                        historic_weight = historic_end.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).minute
                    if future_start.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).hour ==\
                            future_end.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).hour:
                        future_weight = (future_end - future_start).seconds // 60
                    else:
                        # the remainder of the hour.
                        future_weight = 60 - future_start.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).minute
                    # Normalize
                    total_weight = future_weight + historic_weight
                    future_weight /= float(total_weight)
                    historic_weight /= float(total_weight)

                    outside_temperatures[row_time.hour] = future_weight * future_t_out + \
                                                          historic_weight * float(t_out)

                else:
                    outside_temperatures[row_time.hour] = float(t_out)

        # setting outside temperature data for the thermal model.
        thermal_model.set_outside_temperature(outside_temperatures)

        # ===== END: Future and past outside temperature combine =====


        if (cfg["Pricing"]["DR"] and utils.in_between(now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])).time(),
                                                      utils.get_time_datetime(cfg["Pricing"]["DR_Start"]),
                                                      utils.get_time_datetime(cfg["Pricing"]["DR_Finish"]))):  # \
            # or now.weekday() == 4:  # TODO REMOVE ALWAYS HAVING DR ON FRIDAY WHEN DR SUBSCRIBE IS IMPLEMENTED
            DR = True
        else:
            DR = False

        adv_start = time.time()
        adv = Advise([zone],  # array because we might use more than one zone. Multiclass approach.
                     now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])),
                     occ_predictions,
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
                     advise_cfg["Advise"]["Occupancy_Sensors"] if not simulate else False,
                     # TODO Only using config file occupancy for now.
                     safety_constraints)

        action = adv.advise()
        adv_end = time.time()

    except Exception:
        print("ERROR: For zone %s." % zone)
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
    Debugger.debug_print(now, building, zone, adv, safety_constraints, prices, building_setpoints, adv_end - adv_start,
                         file=True)

    # try to commit the changes to the thermostat, if it doesnt work 10 times in a row ignore and try again later
    for i in range(advise_cfg["Advise"]["Thermostat_Write_Tries"]):
        try:
            if not debug and not simulate:
                tstat.write(p)
            # Setting last action in the thermal model after we have succeeded in writing to the tstat.
            thermal_model.set_last_action(
                int(action))
            break
        except:
            if i == advise_cfg["Advise"]["Thermostat_Write_Tries"] - 1:
                e = sys.exc_info()[0]
                print e
                return False, None
            continue

    return True, p


class ZoneThread(threading.Thread):
    def __init__(self, cfg_filename, tstats, zone, client, thermal_model, building, thread_barrier, debug=False,
                 simulate=False, simulate_start=None, simulate_end=None):
        """
        
        :param cfg_filename: 
        :param tstats: 
        :param zone: 
        :param client: 
        :param thermal_model: 
        :param building: 
        :param thread_barrier: threading barrier
        :param debug: Whether to actuate tstats.
        :param simulate: Bool whether to simulate.
        :param simualate_start: (utc datetime) When the simulation should start
        :param simulate_end: (utc datetime) When simulation should end.
        """
        threading.Thread.__init__(self)
        self.cfg_filename = cfg_filename
        self.tstats = tstats
        self.zone = zone
        self.client = client
        self.building = building
        self.thermal_model = thermal_model
        self.debug = debug
        self.thread_barrier = thread_barrier
        self.simulation_results = {"inside": {},
                                   "outside": {},
                                   "heating_setpoint": {},
                                   "cooling_setpoint": {},
                                   "state": {}}
        if self.debug:
            self.simulation_results["noise"] = {}

        # All in utc times
        self.simulate = simulate
        if simulate:
            # Make them utc aware if they are not already.
            self.simulate_start = simulate_start.replace(tzinfo=pytz.utc)
            self.simulate_end = simulate_end.replace(tzinfo=pytz.utc)
            self.simulate_now = self.simulate_start

    def run(self):
        """
        :return: 
        """
        # TODO Rethink how to use the simulate now.
        # TODO MPCLogger is using the current utc time, not the simulation time.


        starttime = time.time()
        action_data = None
        # Run if we are not simulating or if we are simulating, run it until the end is larger than then the now.
        while not self.simulate or (self.simulate and self.simulate_end >= self.simulate_now):
            try:
                # Reloading the config everytime we iterate.
                with open(self.cfg_filename, 'r') as ymlfile:
                    cfg = yaml.load(ymlfile)
                with open("Buildings/" + cfg["Building"] + "/ZoneConfigs/" + self.zone + ".yml", 'r') as ymlfile:
                    advise_cfg = yaml.load(ymlfile)
            except:
                print "There is no " + self.zone + ".yml file under Buildings/" + cfg[
                    "Building"] + "/ZoneConfigs/ folder."
                return  # TODO MAKE THIS RUN NORMAL SCHEDULE SOMEHOW WHEN NO ZONE CONFIG EXISTS

            # Setting now time.
            # The start of this loop is the now time for the process.
            if self.simulate:
                utc_now = self.simulate_now
            else:
                utc_now = utils.get_utc_now()
            cfg_timezone_now = utc_now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"]))

            # check whether to actuate if there is no simulation happening.
            actuate = True
            # TODO have actuate/simulate flag to know what to do. For now, if simulate is true, we will simulate.
            if advise_cfg["Advise"]["Actuate"] and not self.simulate:
                start = advise_cfg["Advise"]["Actuate_Start"]
                end = advise_cfg["Advise"]["Actuate_End"]
                cfg_now_time = cfg_timezone_now.time()
                if utils.in_between(now=cfg_now_time, start=utils.get_time_datetime(start),
                                    end=utils.get_time_datetime(end)):
                    actuate = True
                else:
                    print("Note: We are outside of our actuation regions for zone %s." % self.zone)
                    utils.set_override_false(self.tstats[self.zone])
                    actuate = False
            elif not self.simulate:
                print("WARNING: We are not actuating this zone today. Zone %s is ending." % self.zone)
                return

            # actuate the lights script during the DR event.
            # should happen only once. Hence, one zone is responsible for actuating it.
            # TODO Fix actuate lights with simulation.
            if advise_cfg["Actuate_Lights"] and not self.simulate:
                dr_start = cfg["Pricing"]["DR_Start"]
                dr_end = cfg["Pricing"]["DR_Finish"]
                cfg_now_time = cfg_timezone_now.time()
                if utils.in_between(now=cfg_now_time, start=utils.get_time_datetime(dr_start),
                                    end=utils.get_time_datetime(dr_end)):
                    print("NOTE: Running the lights script from zone %s." % zone)
                    lights.lights(building=cfg["Building"], client=self.client, actuate=True)
                    # Overriding the lights.
                    advise_cfg["Actuate_Lights"] = False
                    with open("Buildings/" + cfg["Building"] + "/ZoneConfigs/" + self.zone + ".yml", 'wb') as ymlfile:
                        yaml.dump(advise_cfg, ymlfile)

            # TODO MASSIVE. FIND DR LAMBDA AND EXPANSION THAT WILL BE USED FOR THE LOGGER.
            if actuate:
                print("Note: Actuating zone %s." % self.zone)

                # Flag whether to run normal schedule. If not, then run MPC.
                go_to_normal_schedule = not advise_cfg["Advise"]["MPC"]

                # Running the MPC if possible.
                if not go_to_normal_schedule:
                    # Run MPC. Try up to advise_cfg["Advise"]["Thermostat_Write_Tries"] to find and write action.
                    count = 0
                    succeeded = False
                    while not succeeded and not go_to_normal_schedule:
                        succeeded, action_data = hvac_control(cfg, advise_cfg, self.tstats, self.client,
                                                              self.thermal_model,
                                                              self.zone, self.building, utc_now,
                                                              debug=self.debug, simulate=self.simulate)

                        # Increment the counter and set the flag for the normal schedule if the MPC failed too often.
                        if not succeeded:
                            time.sleep(10)
                            if count == advise_cfg["Advise"]["Thermostat_Write_Tries"]:
                                print("Problem with MPC, entering normal schedule.")
                                go_to_normal_schedule = True
                            count += 1
                    # Log this action if succeeded.
                    if succeeded:
                        MPCLogger.mpc_log(building, self.zone, utc_now, float(cfg["Interval_Length"]), is_mpc=True,
                                          is_schedule=False,
                                          mpc_lambda=0.995, shut_down_system=False)
                        # TODO Add note in LOGGER that MPC DIDN"T WORK HERE

                # Running the normal Schedule
                if go_to_normal_schedule:
                    # go into normal schedule
                    normal_schedule = NormalSchedule(cfg, self.tstats[self.zone], advise_cfg)

                    # TODO ADD THREAD BARRIER the normal schedule if it needs it for simulation or other stuff.
                    normal_schedule_succeeded, action_data = normal_schedule.normal_schedule(
                        debug=self.debug)  # , simulate=simulate)
                    # Log this action.
                    if normal_schedule_succeeded:
                        MPCLogger.mpc_log(building, self.zone, utc_now, float(cfg["Interval_Length"]), is_mpc=False,
                                          is_schedule=True,
                                          expansion=advise_cfg["Advise"]["Baseline_Dr_Extend_Percent"], shut_down_system=False)
                    else:
                        # If normal schedule fails then we have big problems.

                        # Set the override to false for the thermostat so the local schedules can take over again.
                        utils.set_override_false(tstat)

                        # Logging the shutdown.
                        MPCLogger.mpc_log(self.building, self.zone, utc_now, float(cfg["Interval_Length"]),
                                          is_mpc=False, is_schedule=False, shut_down_system=True,
                                          system_shut_down_msg="Normal Schedule Failed")

                        print("System shutdown for zone %s, normal schedule has not succeeded. "
                              "Returning control to Thermostats. \n" % self.zone)
                        return

            print(
            "This process is for building %s" % cfg["Building"])  # TODO Rethink. now every thread will write this.
            # Wait for the next interval if not simulating.
            if not self.simulate:
                print datetime.datetime.now()
                time.sleep(60. * float(cfg["Interval_Length"]) - (
                    (time.time() - starttime) % (60. * float(cfg["Interval_Length"]))))
            # else, increment the now parameter we have.
            # AND ADVANCE TEMPERATURE OF TSTATS.
            # AND STORE THE SIMULATION RESULTS
            else:
                # NOTE: Making use of the fact that MPCThermalModel stores inside, outside, action data. We
                # will use the action_data from the hvac_control to fill the simulation dictionary as well.
                # Filling simulation_results
                utc_unix_timestamp = time.mktime(self.simulate_now.timetuple())
                self.simulation_results["inside"][str(utc_unix_timestamp)] = self.tstats[self.zone].temperature
                # TODO FIX WEATHER INDEX
                self.simulation_results["outside"][str(utc_unix_timestamp)] = \
                        self.thermal_model.outside_temperature.values()[0]
                self.simulation_results["cooling_setpoint"][str(utc_unix_timestamp)] = action_data["cooling_setpoint"]
                self.simulation_results["heating_setpoint"][str(utc_unix_timestamp)] = action_data["heating_setpoint"]
                self.simulation_results["state"][str(utc_unix_timestamp)] = self.thermal_model.last_action

                print(self.simulate_now)
                self.simulate_now += datetime.timedelta(minutes=cfg["Interval_Length"])
                # advancing the temperatures of each zone. each thread is responsible for this.
                # waiting for all threads to advance their temperatures before proceeding.
                new_temperature, noise = self.tstats[self.zone].next_temperature(debug=self.debug)
                if self.debug:
                    self.simulation_results["noise"][str(utc_unix_timestamp)] = noise
                thread_barrier.wait()

            # Checking if manual setpoint changes occurred.
            if actuate and not self.simulate:
                # end program if setpoints have been changed. (If not writing to tstat we don't want this)
                if action_data is not None and utils.has_setpoint_changed(self.tstats[self.zone], action_data,
                                                                          self.zone, self.building):
                    # Set the override to false for the thermostat so the local schedules can take over again.
                    utils.set_override_false(tstat)

                    # Tell the logger to record this setpoint change.
                    MPCLogger.mpc_log(self.building, self.zone, utils.get_utc_now(), float(cfg["Interval_Length"]),
                                      is_mpc=False, is_schedule=False, shut_down_system=True,
                                      system_shut_down_msg="Manual Setpoint Change")

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

    cfg = utils.get_config(building)

    client = utils.choose_client()  # TODO add config

    hc = HodClient("xbos/hod", client)

    tstats = utils.get_thermostats(client, hc, cfg["Building"])

    # --- Thermal Model Init ------------
    # initialize and fit thermal model

    # only single stage cooling buildings get to retrive data. otherwise takes too long.
    # if building in ["north-berkeley-senior-center", "ciee", "avenal-veterans-hall", "orinda-community-center",
    #                 "avenal-recreation-center", "word-of-faith-cc", "jesse-turner-center", "berkeley-corporate-yard"]:
    if building != "jesse-turner-center":
        thermal_data = utils.get_data(cfg=cfg, client=client, days_back=150, force_reload=False)

        zone_thermal_models = {}
        for zone, zone_data in thermal_data.items():
            # Concat zone data to put all data together and filter such that all datapoints have dt != 1
            filtered_zone_data = zone_data[zone_data["dt"] == 5]
            # print(zone)
            # print(filtered_zone_data.shape)
            if zone != "HVAC_Zone_Please_Delete_Me":
                zone_thermal_models[zone] = MPCThermalModel(zone=zone, thermal_data=filtered_zone_data,
                                                            interval_length=15, thermal_precision=0.05)
    else:
        zone_thermal_models = {}
        for zone in tstats.keys():
            zone_thermal_models[zone] = None

    print("Trained Thermal Model")
    # --------------------------------------

    # TODO most likely through datamanager or some sort of input. Get current temperatures for simulation
    curr_temperatures_simulation = {zone: dict_tstat.temperature for zone, dict_tstat in tstats.items()}

    # TODO change simulate
    simulate = False

    # Getting the simulateTstats
    simulate_tstats = {}
    if simulate:
        for zone, thermal_model_zone in zone_thermal_models.items():
            zone_simulate_tstat = SimulationTstat(thermal_model_zone, curr_temperatures_simulation[zone])
            zone_simulate_tstat.set_gaussian_distributions(thermal_data[zone], thermal_data[zone]["t_next"])
            simulate_tstats[zone] = zone_simulate_tstat

    # choose the tstats to use.
    if simulate:
        mpc_tstats = simulate_tstats
    else:
        mpc_tstats = tstats

    # simulate_start = datetime.datetime(year=2018, month=4, day=3)
    # simulate_end = simulate_start + datetime.timedelta(hours=1)
    simulate_start = utils.get_utc_now() - datetime.timedelta(hours=0.5)
    simulate_end = simulate_start + datetime.timedelta(hours=1)

    # setting a barrier for the threads. this will be used in hvac_control for stopping all threads before setting
    # all zone_temperatures and weather for the thermalModel
    num_threads = len(tstats.keys())
    thread_barrier = utils.Barrier(num_threads)
    threads = []

    for zone, tstat in mpc_tstats.items():
        # TODO only because we want to only run on the basketball courts.
        if building != "jesse-turner-center" or "Basketball" in zone:
            thread = ZoneThread(yaml_filename, mpc_tstats, zone, client, zone_thermal_models[zone],
                                cfg["Building"], thread_barrier, debug=True, simulate=simulate,
                                simulate_start=simulate_start, simulate_end=simulate_end)
            thread.start()
            threads.append(thread)

    for t in threads:
        t.join()

    for t in threads:
        print(t.zone)
        print(t.simulation_results)
