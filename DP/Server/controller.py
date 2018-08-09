import datetime
import math
import sys
import threading
import time
import traceback

import pytz

import utils
from IteratedDP import IteratedDP
from LinearProgram import AdviseLinearProgram
from NormalSchedule import NormalSchedule
from SimulationTstat import SimulationTstat

sys.path.insert(0, './MPC')
sys.path.insert(0, './MPC/ThermalModels')
from MPCThermalModel import MPCThermalModel

# from AverageThermalModel import *

sys.path.insert(0, '../Utils')
sys.path.append("./Lights")
import MPCLogger

from xbos.services.hod import HodClient


# TODO set up a moving average for how long it took for action to take place.
# the main controller
def hvac_control(cfg_building, cfg_zones, tstats, thermal_model, zones, building, start, consumption_storage,
                 actuate_zones, debug=False, optimization_type="DP", dp_optimizer=None, lp_optimizer=None):
    """
    
    :param cfg_building: building config.
    :param cfg_zones: {zone: cfg_zone}
    :param tstats:
    :param thermal_model:
    :param zones: (list str) list of zones for which to run hvac control
    :param start: datetime object in UTC which tells the control what now is.
    :param debug: wether to actuate the tstat.
    :param actuate_zones: {zone: boolean} whether to actuate the zone.
    :param optimization_type: str "DP" for dynamic programming approach or "LP" for Linear Programing 
    :return: boolean, dict. Success Boolean indicates whether writing action has succeeded. Dictionary {cooling_setpoint: float,
    heating_setpoint: float, override: bool, mode: int} and None if success boolean is flase.
    """
    # TODO Fix actuate variable because now if we are not acutating we are not using occupancy predictions.

    # I dont like this.
    end = start + datetime.timedelta(hours=4)

    try:

        zone_temperatures = {dict_zone: dict_tstat.temperature for dict_zone, dict_tstat in tstats.items()}

        adv_start = time.time()

        # Start given optimization
        if optimization_type == "DP":
            optimal_actions = dp_optimizer.advise(start, end, zones, cfg_building, cfg_zones, thermal_model,
                                         zone_temperatures, consumption_storage)
        elif optimization_type == "LP":
            optimal_actions = lp_optimizer.advise(start, end, zones, cfg_building, cfg_zones, thermal_model,
                                         zone_temperatures, consumption_storage)
        else:
            raise Exception("Invalid optimization type given.")

        if debug:
            print("Action for zones %s" % str(optimal_actions))
            print("")

        adv_end = time.time()

    except Exception:
        print("ERROR: For zones %s." % str(zones))
        print(traceback.format_exc())
        # TODO Find a better way for exceptions
        return False, None

    # Get data for finding setpoints given the action
    # TODO FIX INTERVAL
    INTERVAL = 15
    # TODO FIX The TIME in utils. it returns data which has second and microsecond set to zero.
    start_with_striped = start.replace(microsecond=0, second=0)
    end_with_striped = end.replace(microsecond=0, second=0)
    safety_constraints_zones = utils.get_safety_matrix(building, zones, start_with_striped, end_with_striped, INTERVAL)

    # Initialize dictionary which keeps track of the success of writing and finding setpoints for the given zone.
    succeeded_zones = {}
    # Initialize dictionary which keeps track of the messages which should be written to zones.
    message_zones = {}

    # Find the setpoints for each zone given the actions and actuating zone if necessary.

    for iter_zone in zones:
        # Get zone specific information
        cfg_zone = cfg_zones[iter_zone]
        temperature_zone = zone_temperatures[iter_zone]
        action_zone = optimal_actions[iter_zone]
        safety_constraint_current = safety_constraints_zones[iter_zone].loc[start_with_striped]
        tstat = tstats[iter_zone]

        # action "0" is Do Nothing, action "1" is Heating, action "2" is Cooling
        if action_zone == utils.NO_ACTION:
            heating_setpoint = temperature_zone - cfg_zone["Advise"]["Minimum_Comfortband_Height"] / 2.
            cooling_setpoint = temperature_zone + cfg_zone["Advise"]["Minimum_Comfortband_Height"] / 2.

            if heating_setpoint < safety_constraint_current["t_low"]:
                heating_setpoint = safety_constraint_current["t_low"]

                if (cooling_setpoint - heating_setpoint) < cfg_zone["Advise"]["Minimum_Comfortband_Height"]:
                    cooling_setpoint = min(safety_constraint_current["t_high"],
                                           heating_setpoint + cfg_zone["Advise"]["Minimum_Comfortband_Height"])

            elif cooling_setpoint > safety_constraint_current["t_high"]:
                cooling_setpoint =safety_constraint_current["t_high"]

                if (cooling_setpoint - heating_setpoint) < cfg_zone["Advise"]["Minimum_Comfortband_Height"]:
                    heating_setpoint = max(safety_constraint_current["t_high"]["t_low"],
                                           cooling_setpoint - cfg_zone["Advise"]["Minimum_Comfortband_Height"])

            # round to integers since the thermostats round internally.
            heating_setpoint = math.floor(heating_setpoint)
            cooling_setpoint = math.ceil(cooling_setpoint)

            p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint,
                 "mode": 3}
            print "Doing nothing"

        # TODO Rethink how we set setpoints for heating and cooling and for DR events.
        # heating
        elif action_zone == utils.HEATING_ACTION:
            heating_setpoint = temperature_zone + 2 * cfg_zone["Advise"]["Hysterisis"]
            cooling_setpoint = heating_setpoint + cfg_zone["Advise"]["Minimum_Comfortband_Height"]

            if cooling_setpoint > safety_constraint_current["t_high"]:
                cooling_setpoint = safety_constraint_current["t_high"]

                # making sure we are in the comfortband
                if (cooling_setpoint - heating_setpoint) < cfg_zone["Advise"]["Minimum_Comfortband_Height"]:
                    heating_setpoint = max(safety_constraint_current["t_low"],
                                           cooling_setpoint - cfg_zone["Advise"]["Minimum_Comfortband_Height"])

            # round to integers since the thermostats round internally.
            heating_setpoint = math.ceil(heating_setpoint)
            cooling_setpoint = math.ceil(cooling_setpoint)

            p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint,
                 "mode": 3}
            print "Heating"

        # cooling
        elif action_zone == utils.COOLING_ACTION:
            cooling_setpoint = temperature_zone - 2 * cfg_zone["Advise"]["Hysterisis"]
            heating_setpoint = cooling_setpoint - cfg_zone["Advise"]["Minimum_Comfortband_Height"]

            if heating_setpoint < safety_constraint_current["t_low"]:
                heating_setpoint = safety_constraint_current["t_low"]

                # making sure we are in the comfortband
                if (cooling_setpoint - heating_setpoint) < cfg_zone["Advise"]["Minimum_Comfortband_Height"]:
                    cooling_setpoint = min(safety_constraint_current["t_high"],
                                           heating_setpoint + cfg_zone["Advise"]["Minimum_Comfortband_Height"])

            # round to integers since the thermostats round internally.
            heating_setpoint = math.floor(heating_setpoint)
            cooling_setpoint = math.floor(cooling_setpoint)

            p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint,
                 "mode": 3}
            print "Cooling"
        else:
            print "Problem with action."
            succeeded_zones[iter_zone] = False
            message_zones[iter_zone] = None

        print("Zone: " + iter_zone + ", action: " + str(p))

        # # Plot the MPC graph.
        # if advise_cfg["Advise"]["Print_Graph"]:
        #     adv.g_plot(iter_zone)

        # # Log the information related to the current MPC
        # Debugger.debug_print(now, building, iter_zone, adv, safety_constraints, prices, building_setpoints, adv_end - adv_start,
        #                      file=True)

        # try to commit the changes to the thermostat, if it doesnt work 10 times in a row ignore and try again later
        for i in range(cfg_zone["Advise"]["Thermostat_Write_Tries"]):
            try:
                if actuate_zones[iter_zone]:
                    tstat.write(p)
                # Setting last action in the thermal model after we have succeeded in writing to the tstat.
                zone_thermal_models[iter_zone].set_last_action(
                    int(action_zone))
                break
            except:
                if i == cfg_zone["Advise"]["Thermostat_Write_Tries"] - 1:
                    print("Could not write to tstat for zone %s. Trying again." % iter_zone)
                    succeeded_zones[iter_zone] = False
                    message_zones[iter_zone] = None
                continue

        succeeded_zones[iter_zone] = True
        message_zones[iter_zone] = p

    return succeeded_zones, message_zones


class ZoneThread(threading.Thread):
    def __init__(self, tstats, zones, client, thermal_model, building, barrier, mutex, consumption_storage,
                 optimization_type, dp_optimizer=None, lp_optimizer=None, debug=False,
                 simulate=False, simulate_start=None, simulate_end=None):
        """
        
        :param tstats: 
        :param zone: 
        :param client: 
        :param thermal_model: 
        :param building: 
        :param barrier: threading barrier. Needs to be shared by threads. 
        :param mutex: threading mutex. Needs to be shared by threads. 
        :param consumption_storage: ConsumptionStorage object shared across threads. 
        :param optimization_type: str "DP" for dynamic programming and "LP" for Linear Programming
        :param dp_optimizer: DP optimizer class which is shared by all threads.
        :param lp_optimizer: LP Optimizer class which is shared by all threads. 
        :param debug: Whether to actuate tstats.
        :param simulate: Bool whether to simulate.
        :param simualate_start: (utc datetime) When the simulation should start
        :param simulate_end: (utc datetime) When simulation should end.
        """
        threading.Thread.__init__(self)
        self.tstats = tstats
        self.zones = zones
        self.client = client
        self.building = building
        self.thermal_model = thermal_model
        self.debug = debug
        self.barrier = barrier
        self.mutex = mutex
        self.simulation_results = {"inside": {},
                                   "outside": {},
                                   "heating_setpoint": {},
                                   "cooling_setpoint": {},
                                   "state": {}}
        if self.debug:
            self.simulation_results["noise"] = {}

        # setting consumption storage
        self.consumption_storage = consumption_storage

        # Checking if valid optimization parameters were given
        if optimization_type == "DP" and dp_optimizer is None:
            raise Exception("ERROR: Wanting to optimize with DP but did not give DP class.")
        if optimization_type == "LP" and lp_optimizer is None:
            raise Exception("ERROR: Wanting to optimize with LP but did not give LP class.")
        if optimization_type not in ["DP", "LP"]:
            raise Exception("ERROR: Did not give a valid optimization type argument.")

        # setting optimizers which are shared between threads.
        self.dp_optimizer = dp_optimizer
        self.lp_optimizer = lp_optimizer
        self.optimization_type = optimization_type

        # All in utc times
        self.simulate = simulate
        if simulate:
            # Make them utc aware if they are not already.
            self.simulate_start_utc = simulate_start.replace(tzinfo=pytz.utc)
            self.simulate_end_utc = simulate_end.replace(tzinfo=pytz.utc)

    def run(self):
        """
        :return: 
        """
        # TODO Rethink how to use the simulate now.
        # TODO MPCLogger is using the current utc time, not the simulation time.


        starttime = time.time()
        action_data = None

        # Setting now time before the loop. From here on the now time will be set at the end of the loop.
        if self.simulate:
            now_utc = self.simulate_start_utc
        else:
            now_utc = utils.get_utc_now()

        # ======== Start Program =========
        # do-while loop with check after getting start/end times for the program.
        # Initializing the dictionary which contains the flag of whether to stop program for the given zone.
        # Loop will be stopped when all zones have the flag set to true.
        stop_program_zone = {iter_zone: False for iter_zone in self.zones}
        while True:
            # Reloading the config every time we iterate.
            try:
                cfg_building = utils.get_config(self.building)
                cfg_zones = {iter_zone: utils.get_zone_config(self.building, iter_zone) for iter_zone in self.zones}
                # TODO uncomment once we shifted to multiple zones for one thread.
                # cfg_zone = {}
                # for iter_zone in self.zones:
                #     cfg_zone[iter_zone] = utils.get_zone_config(self.building, iter_zone)
            except:
                print "There is no " + str(self.zones) + ".yml file under Buildings/" + cfg_building[
                    "Building"] + "/ZoneConfigs/ folder."
                return  # TODO MAKE THIS RUN NORMAL SCHEDULE SOMEHOW WHEN NO ZONE CONFIG EXISTS It will raise an error for now

            # Getting timezone from the config
            timezone_cfg = utils.get_config_timezone(cfg_building)

            # Adjust the time zone of the now time.
            now_cfg_timezone = now_utc.astimezone(tz=pytz.timezone(cfg_building["Pytz_Timezone"]))

            # dictionary to store the start and end times of actuation for each zone.
            actuation_start_end_cfg_tz = {}

            # setting when the mpc/normal should start and end for actuation and simulation.
            for iter_zone, cfg_zone in cfg_zones.items():
                if self.simulate:
                    # TODO This should be replaced with getting this data from config files.
                    start_cfg_timezone = self.simulate_start_utc.astimezone(tz=timezone_cfg)
                    end_cfg_timezone = self.simulate_end_utc.astimezone(tz=timezone_cfg)
                else:
                    # For now we will leave the config file start and end times for the current day we are running
                    start_cfg_string = cfg_zone["Advise"]["Actuate_Start"]
                    end_cfg_string = cfg_zone["Advise"]["Actuate_End"]
                    start_cfg_timezone = utils.combine_date_time(start_cfg_string, now_cfg_timezone)
                    end_cfg_timezone = utils.combine_date_time(end_cfg_string, now_cfg_timezone)

                # Edge case where the end time may be at midnight and hence start time > end time
                if (start_cfg_timezone > end_cfg_timezone) and \
                        (end_cfg_timezone.minute == 0 and end_cfg_timezone.hour == 0):
                    end_cfg_timezone += datetime.timedelta(days=1)

                # add start and end times to actuation_start_end_cfg_tz.
                actuation_start_end_cfg_tz[iter_zone] = [start_cfg_timezone, end_cfg_timezone]

            # The test for whether to stop the loop. Dictionary for which zones to not run the program anymore.
            stop_program_zone = {iter_zone: start_end[1] < now_cfg_timezone
                                            or stop_program_zone[iter_zone]
                                 for iter_zone, start_end in actuation_start_end_cfg_tz.items()}

            # check whether to actuate if there is no simulation happening. Don't actuate if we would like to terminate
            # the program for the given zone.
            actuate_zones = {iter_zone: (start_end[0] <= now_cfg_timezone <= start_end[1])
                                   and (not self.simulate)
                                   and not stop_program_zone[iter_zone] for
                             iter_zone, start_end in actuation_start_end_cfg_tz.items()}

            # debug messages for start and end times
            if self.debug:
                self.mutex.acquire()
                for zone, start_end in actuation_start_end_cfg_tz.items():
                    print("----- For zones %s. -----" % zone)
                    print("Now time in building timezone: %s" % utils.get_datetime_to_string(now_cfg_timezone))
                    print("Start time in building timezone: %s" % utils.get_datetime_to_string(start_end[0]))
                    print("End time in building timezone: %s" % utils.get_datetime_to_string(start_end[1]))
                    print("Should actuate", actuate_zones[zone])
                    print("Is past endtime", stop_program_zone[zone])
                    print("Is simulating", self.simulate)
                    print("")
                self.mutex.release()

            # Stop the loop if all zones have flag set.
            if all(stop_program_zone.values()):
                self.mutex.acquire()
                print("Stopping program for zones %s due to being past actuation end time or manual setpoint "
                      "changes." % str(self.zones))
                print("")
                self.mutex.release()
                break

            # actuate the lights script during the DR event.
            # should happen only once. Hence, only one thread is responsible for actuating it. It's the one which
            # acquires the mutex firest.
            # TODO Fix actuate lights with simulation and multizones.
            # self.mutex.acquire()
            # if cfg_zone["Actuate_Lights"] and not self.simulate:
            #     utils.actuate_lights(now_cfg_timezone, cfg_building, cfg_zone, self.zone, self.client)
            # self.mutex.release()

            # TODO MASSIVE. FIND DR LAMBDA AND EXPANSION THAT WILL BE USED FOR THE LOGGER.
            if any(actuate_zones.values()) or self.simulate:

                # Flag whether to run MPC.
                run_mpc_zones = {iter_zone: cfg_zone["Advise"]["MPC"] for iter_zone, cfg_zone in cfg_zones.items()}

                # Running the MPC if possible.
                if any(run_mpc_zones.values()):
                    # Run MPC. Try up to cfg_zone["Advise"]["Thermostat_Write_Tries"] to find and write action.
                    try_count = 0
                    # set that all zones have not succeeded yet.
                    succeeded_zones = {iter_zone: False for iter_zone in self.zones}
                    while not all(succeeded_zones.values()) and any(run_mpc_zones.values()):
                        # TODO (FOR NOW DEBUG ALSO MAKES THE PROGRAM NOT ACTUATE)
                        succeeded_zones, action_data = hvac_control(cfg_building, cfg_zones, self.tstats,
                                                              self.thermal_model,
                                                              self.zones, self.building, now_utc,
                                                              consumption_storage=self.consumption_storage,
                                                              debug=self.debug, actuate_zones=actuate_zones,
                                                              optimization_type=self.optimization_type,
                                                              dp_optimizer=self.dp_optimizer,
                                                              lp_optimizer=self.lp_optimizer)

                        # Increment the counter and set the flag for the normal schedule if the MPC failed too often.
                        if not all(succeeded_zones.values()):
                            # Add once since we tried once and didn't succeed.
                            try_count += 1
                            time.sleep(10)
                            for iter_zone, cfg_zone in cfg_zones.items():
                                if try_count == cfg_zone["Advise"]["Thermostat_Write_Tries"]:
                                    self.mutex.acquire()
                                    print("Problem with MPC for zone %s, entering normal schedule." % iter_zone)
                                    print("")
                                    self.mutex.release()
                                    run_mpc_zones[iter_zone] = False
                                    # TODO Add note in LOGGER that MPC DIDN"T WORK HERE
                            # Set actuate to False for zones that succeeded. No need to actuate again till next round.
                            for iter_zone, succeeded in succeeded_zones.items():
                                if succeeded:
                                    actuate_zones[iter_zone] = False

                    # Log this action if succeeded.
                    if all(succeeded_zones.values()):
                        # TODO If we are in an interval that overlaps with the end of the DR event then
                        # part of the optimization for one interval is in DR and the other outside. We should account
                        # for this.
                        print("SHOULD LOG HERE")
                        for iter_zone, cfg_zone in cfg_zones.items():
                            is_dr = utils.is_DR(now_cfg_timezone, cfg_building)
                            if is_dr:
                                mpc_lambda = cfg_zone["Advise"]["DR_Lambda"]
                            else:
                                mpc_lambda = cfg_zone["Advise"]["General_Lambda"]

                            # TODO update MPCLogger to multizones.
                            MPCLogger.mpc_log(building, iter_zone, now_utc, float(cfg_building["Interval_Length"]),
                                              is_mpc=True,
                                              is_schedule=False,
                                              mpc_lambda=mpc_lambda, shut_down_system=False)

                # Running the normal Schedule
                if not all(run_mpc_zones.values()):
                    # go into normal schedule
                    normal_schedule = NormalSchedule(cfg_building, self.tstats[self.zone], cfg_zone)

                    # TODO ADD THREAD BARRIER the normal schedule if it needs it for simulation or other stuff.
                    normal_schedule_succeeded, action_data = normal_schedule.normal_schedule(
                        debug=self.debug)  # , simulate=simulate)
                    # Log this action.
                    if normal_schedule_succeeded:
                        MPCLogger.mpc_log(building, self.zone, now_utc, float(cfg_building["Interval_Length"]),
                                          is_mpc=False,
                                          is_schedule=True,
                                          expansion=cfg_zone["Advise"]["Baseline_Dr_Extend_Percent"],
                                          shut_down_system=False)
                    else:
                        # If normal schedule fails then we have big problems.

                        # Set the override to false for the thermostat so the local schedules can take over again.
                        utils.set_override_false(tstat)

                        # Logging the shutdown.
                        MPCLogger.mpc_log(self.building, self.zone, now_utc, float(cfg_building["Interval_Length"]),
                                          is_mpc=False, is_schedule=False, shut_down_system=True,
                                          system_shut_down_msg="Normal Schedule Failed")

                        print("System shutdown for zone %s, normal schedule has not succeeded. "
                              "Returning control to Thermostats. \n" % self.zone)
                        return

            print(
                "This process is for building %s" % cfg_building[
                    "Building"])  # TODO Rethink. now every thread will write this.

            # Wait for the next interval if not simulating.
            if not self.simulate:
                time.sleep(60. * float(cfg_building["Interval_Length"]) - (
                    (time.time() - starttime) % (60. * float(cfg_building["Interval_Length"]))))

                # set now time.
                now_utc = utils.get_utc_now()

                # TODO MAYBE SET NOW TIME HERE.
            # else, increment the now parameter we have.
            # AND ADVANCE TEMPERATURE OF TSTATS.
            # AND STORE THE SIMULATION RESULTS
            else:
                # NOTE: Making use of the fact that MPCThermalModel stores inside, outside, action data. We
                # will use the action_data from the hvac_control to fill the simulation dictionary as well.
                # Filling simulation_results
                utc_unix_timestamp = time.mktime(now_utc.timetuple())
                self.simulation_results["inside"][str(utc_unix_timestamp)] = self.tstats[self.zone].temperature
                # TODO FIX WEATHER INDEX
                self.simulation_results["outside"][str(utc_unix_timestamp)] = \
                    self.thermal_model.outside_temperature.values()[0]
                self.simulation_results["cooling_setpoint"][str(utc_unix_timestamp)] = action_data["cooling_setpoint"]
                self.simulation_results["heating_setpoint"][str(utc_unix_timestamp)] = action_data["heating_setpoint"]
                self.simulation_results["state"][str(utc_unix_timestamp)] = self.thermal_model.last_action

                # advancing the temperatures of each zone. each thread is responsible for this.
                # waiting for all threads to advance their temperatures before proceeding.
                new_temperature, noise = self.tstats[self.zone].next_temperature(debug=self.debug)
                if self.debug:
                    self.simulation_results["noise"][str(utc_unix_timestamp)] = noise
                self.barrier.wait()

                # set now time.
                now_utc += datetime.timedelta(minutes=cfg_building["Interval_Length"])

            # Checking if manual setpoint changes occurred given that we actuated that zone.
            for iter_zone, actuate_zone in actuate_zones.items():
                if actuate_zone:
                    # end program if setpoints have been changed. (If not writing to tstat we don't want this)
                    if action_data is not None and utils.has_setpoint_changed(self.tstats[iter_zone], action_data,
                                                                              iter_zone, self.building):
                        # Tell the logger to record this setpoint change.
                        MPCLogger.mpc_log(self.building, iter_zone, utils.get_utc_now(),
                                          float(cfg_building["Interval_Length"]),
                                          is_mpc=False, is_schedule=False, shut_down_system=True,
                                          system_shut_down_msg="Manual Setpoint Change")
                        print("Ending program for zone %s due to manual setpoint changes. \n" % iter_zone)
                        # Do not actuate the zone again till the end of the program
                        stop_program_zone[iter_zone] = True

        # TODO FIX FIX FIX
        # if actuate:
        #     # TODO FIX THE UPDATE STEP. PUT THIS OUTSIDE OF HVAC CONTROL. Put where we wait and then set now time if
        #     TODO we are not simulating.
        #     # NOTE: call update before setWeatherPredictions and set_temperatures
        #     thermal_model.update(zone_temperatures, interval=cfg["Interval_Length"])

        if not self.simulate:
            # Set the override to false for the thermostat so the local schedules can take over again.
            utils.set_override_false(tstat)


if __name__ == '__main__':
    # TODO check for comfortband height and whether correctly implemented
    building = sys.argv[1]
    ZONES = utils.get_zones(building)

    # TODO change simulate and debug.
    debug = True
    simulate = True

    # read from config file
    try:
        yaml_filename = "Buildings/%s/%s.yml" % (sys.argv[1], sys.argv[1])
    except:
        sys.exit("Please specify the configuration file as: python2 controller.py config_file.yaml")

    cfg_building = utils.get_config(building)

    if debug:
        client = utils.choose_client()
    else:
        client = utils.choose_client(cfg_building)

    hc = HodClient("xbos/hod", client)

    tstats = utils.get_thermostats(client, hc, cfg_building["Building"])

    # --- Thermal Model Init ------------
    # initialize and fit thermal model

    # only single stage cooling buildings get to retrive data. otherwise takes too long.
    # if building in ["north-berkeley-senior-center", "ciee", "avenal-veterans-hall", "orinda-community-center",
    #                 "avenal-recreation-center", "word-of-faith-cc", "jesse-turner-center", "berkeley-corporate-yard"]:
    if building != "jesse-turner-center":
        thermal_data = utils.get_data(cfg=cfg_building, client=client, days_back=150, force_reload=False)

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


    dp_optimizer = IteratedDP(building, client, len(mpc_tstats.keys()), num_iterations=1)
    lp_optimizer = AdviseLinearProgram(building, client, num_threads=1, debug=False)
    consumption_storage = None

    optimization_type = "LP"

    if optimization_type == "DP":
        # setting a barrier for the threads. this will be used in hvac_control for stopping all threads before setting
        # all zone_temperatures and weather for the thermalModel
        num_threads = len(tstats.keys())
        thread_barrier = utils.Barrier(num_threads)
        thread_mutex = threading.Semaphore(1)
        threads = []
        for zone, tstat in mpc_tstats.items():
            # TODO only because we want to only run on the basketball courts.
            if building != "jesse-turner-center" or "Basketball" in zone:
                thread = ZoneThread(mpc_tstats, [zone], client, zone_thermal_models[zone],
                                    cfg_building["Building"], thread_barrier, thread_mutex,
                                    consumption_storage=consumption_storage,
                                    optimization_type=optimization_type, dp_optimizer=dp_optimizer,
                                    lp_optimizer=lp_optimizer,
                                    debug=debug, simulate=simulate,
                                    simulate_start=simulate_start, simulate_end=simulate_end)
                thread.start()
                threads.append(thread)

    elif optimization_type == "LP":
        # setting a barrier for the threads. this will be used in hvac_control for stopping all threads before setting
        # all zone_temperatures and weather for the thermalModel
        thread_barrier = utils.Barrier(1)
        thread_mutex = threading.Semaphore(1)
        threads = []
        thread = ZoneThread(mpc_tstats, ZONES, client, zone_thermal_models,
                            cfg_building["Building"], thread_barrier, thread_mutex,
                            consumption_storage=consumption_storage,
                            optimization_type=optimization_type, dp_optimizer=dp_optimizer,
                            lp_optimizer=lp_optimizer,
                            debug=debug, simulate=simulate,
                            simulate_start=simulate_start, simulate_end=simulate_end)
        thread.start()
        threads.append(thread)

    else:
        raise Exception("ERROR: Invalid Optimization type given.")

    for t in threads:
        t.join()

    for t in threads:
        print(t.zones)
        print(t.simulation_results)
