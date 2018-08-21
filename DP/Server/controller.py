import datetime
import sys
import threading
import time
from collections import defaultdict

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
                 actuate_zones, optimizer, debug=False, client=None):
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

    # ---- Optimization ----
    zone_temperatures = {dict_zone: dict_tstat.temperature for dict_zone, dict_tstat in tstats.items()}

    adv_start = time.time()

    # Start given optimization
    optimal_actions, optimal_setpoints = optimizer.advise(start, end, zones, cfg_building, cfg_zones, thermal_model,
                                                          zone_temperatures, consumption_storage)

    if debug:
        print("Action for zones %s" % str(optimal_actions))
        print("")

    adv_end = time.time()

    # ---- Optimization end ----


    # Get data for finding setpoints given the action
    # TODO FIX INTERVAL
    INTERVAL = 15
    # TODO FIX The TIME in utils. it returns data which has second and microsecond set to zero.
    # start_with_striped = start.replace(microsecond=0, second=0)
    # end_with_striped = end.replace(microsecond=0, second=0)
    safety_constraints_zones = utils.get_safety_matrix(building, zones, start, end, INTERVAL,
                                                       datamanager_zones=utils.get_zone_data_managers(building, zones,
                                                                                                      start, client))

    # Initialize dictionary which keeps track of the success of writing and finding setpoints for the given zone.
    succeeded_zones = {}
    # Initialize dictionary which keeps track of the messages which should be written to zones.
    message_zones = {}

    # Find the setpoints for each zone given the actions and actuating zone if necessary.

    for iter_zone in zones:
        # Get zone specific information
        cfg_zone = cfg_zones[iter_zone]
        temperature_zone = zone_temperatures[iter_zone]
        safety_constraint_current = safety_constraints_zones[iter_zone].loc[
            start.astimezone(tz=utils.get_config_timezone(cfg_building))]
        tstat = tstats[iter_zone]

        # set the setpoints we want to use
        if optimal_actions is not None:
            optimal_setpoint = utils.action_logic(cfg_zone, temperature_zone, optimal_actions[iter_zone])
        else:
            assert optimal_setpoints is not None
            optimal_setpoint = optimal_setpoints[iter_zone]

        msg = utils.safety_check(cfg_zone, temperature_zone, safety_constraint_current,
                                 optimal_setpoint["cooling_setpoint"], optimal_setpoint["heating_setpoint"])

        # Set additional information for tstat message
        msg["override"] = True
        msg["mode"] = 3

        print("Zone: " + iter_zone + ", action: " + str(msg))

        # # Plot the MPC graph.
        # if advise_cfg["Advise"]["Print_Graph"]:
        #     adv.g_plot(iter_zone)

        # # Log the information related to the current MPC
        # Debugger.debug_print(now, building, iter_zone, adv, safety_constraints, prices, building_setpoints, adv_end - adv_start,
        #                      file=True)

        # try to commit the changes to the thermostat, if it doesnt work 10 times in a row ignore and try again later
        if msg is not None:
            for i in range(cfg_zone["Advise"]["Thermostat_Write_Tries"]):
                try:
                    if actuate_zones[iter_zone] and not debug:
                        tstat.write(msg)

                    succeeded_zones[iter_zone] = True
                    message_zones[iter_zone] = msg
                    break
                except:
                    if i == cfg_zone["Advise"]["Thermostat_Write_Tries"] - 1:
                        print("Could not write to tstat for zone %s. Trying again." % iter_zone)
                        succeeded_zones[iter_zone] = False
                        message_zones[iter_zone] = None
                    continue
        else:
            # We qualify not getting a good action as an unsuccessful optimization.
            succeeded_zones[iter_zone] = False
            message_zones[iter_zone] = None

    return succeeded_zones, message_zones


def get_action_from_setpoints(cfg_zone, setpoints, temperature_zone):
    cooling_setpoint = setpoints["cooling_setpoint"]
    heating_setpoint = setpoints["heating_setpoint"]

    hysterisis = cfg_zone["Advise"]["Hysterisis"]

    if cooling_setpoint < temperature_zone + hysterisis:
        return utils.COOLING_ACTION
    elif heating_setpoint > temperature_zone + hysterisis:
        return utils.HEATING_ACTION
    else:
        return utils.NO_ACTION


class ZoneThread(threading.Thread):
    def __init__(self, tstats, zones, client, thermal_models, building, barrier, mutex, consumption_storage,
                 normal_schedule,
                 optimization_type, dp_optimizer=None, lp_optimizer=None, debug=False,
                 simulate=False, simulate_start=None, simulate_end=None):
        """
        
        :param tstats: 
        :param zone: 
        :param client: 
        :param thermal_models: {zone: thermal_model}
        :param building: 
        :param barrier: threading barrier. Needs to be shared by threads. 
        :param mutex: threading mutex. Needs to be shared by threads. 
        :param consumption_storage: ConsumptionStorage object shared across threads. 
        :param normal_schedule: The normal schedule class that should be shared with the zones in this thread.
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
        self.thermal_models = thermal_models
        self.debug = debug
        self.barrier = barrier
        self.mutex = mutex
        if not self.debug:
            self.simulation_results = defaultdict(lambda: {"inside": {},
                                                           "outside": {},
                                                           "heating_setpoint": {},
                                                           "cooling_setpoint": {},
                                                           "state": {}})
        else:
            self.simulation_results = defaultdict(lambda: {"inside": {},
                                                           "outside": {},
                                                           "heating_setpoint": {},
                                                           "cooling_setpoint": {},
                                                           "state": {},
                                                           "noise": {}})

        # setting consumption storage
        self.consumption_storage = consumption_storage

        # setting normal schedule
        self.normal_schedule = normal_schedule

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

            # check whether to actuate if there is no simulation happening. Don't actuate if we would like to stop
            # the program for the given zone.
            actuate_zones = {iter_zone: (start_end[0] <= now_cfg_timezone <= start_end[1])
                                        and (not self.simulate)
                                        and not stop_program_zone[iter_zone] for
                             iter_zone, start_end in actuation_start_end_cfg_tz.items()}

            # Keeps track of which zones where the optimization and normal schedule worked (weather for simulation or
            # normal acutation).
            did_succeed_zones = {iter_zone: False for iter_zone in self.zones}

            # set variable to store message data for each zone.
            message_data_zones = {}

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

                # Set a dictionary for zones that need to be actuated but have not yet been successful
                # and which can be actuated through the MPC.
                should_actuate_zones_mpc = {iter_zone: actuate_zones[iter_zone] and not did_succeed_zones[iter_zone]
                                                       and run_mpc_zones[iter_zone]
                                            for iter_zone in self.zones}

                # Running the MPC if possible.
                if any(run_mpc_zones.values()):
                    # Run MPC. Try up to cfg_zone["Advise"]["Thermostat_Write_Tries"] to find and write action.
                    try_count_zones = {iter_zone: 0 for iter_zone in self.zones}

                    # set that all zones have not succeeded yet.
                    succeeded_zones = {iter_zone: False for iter_zone in self.zones}

                    while not all(succeeded_zones.values()) and any(run_mpc_zones.values()):
                        if self.optimization_type == "DP":
                            succeeded_zones, action_data = hvac_control(cfg_building, cfg_zones, self.tstats,
                                                                        self.thermal_models,
                                                                        self.zones, self.building, now_utc,
                                                                        consumption_storage=self.consumption_storage,
                                                                        debug=self.debug,
                                                                        actuate_zones=should_actuate_zones_mpc,
                                                                        optimizer=self.dp_optimizer, client=self.client)
                        elif self.optimization_type == "LP":
                            succeeded_zones, action_data = hvac_control(cfg_building, cfg_zones, self.tstats,
                                                                        self.thermal_models,
                                                                        self.zones, self.building, now_utc,
                                                                        consumption_storage=self.consumption_storage,
                                                                        debug=self.debug,
                                                                        actuate_zones=should_actuate_zones_mpc,
                                                                        optimizer=self.lp_optimizer, client=self.client)

                        # Increment the counter and set the flag for the normal schedule if the MPC failed too often.
                        # Add one since we tried once and didn't succeed.
                        for iter_zone, iter_zone_succeeded in succeeded_zones.items():
                            if not iter_zone_succeeded:
                                try_count_zones[iter_zone] += 1
                            else:
                                # If succeeded then we actuated and don't need to actuate any more.
                                # Also, set setpoints that were executed.
                                message_data_zones[iter_zone] = action_data[iter_zone]
                                should_actuate_zones_mpc[iter_zone] = False
                                did_succeed_zones[iter_zone] = True

                        # If not all succeeded, then we should wait and try again. If it failed too often, normal
                        # schedule needs to be called.
                        if not all(succeeded_zones.values()):
                            time.sleep(10)
                            for iter_zone, cfg_zone in cfg_zones.items():
                                if try_count_zones[iter_zone] == cfg_zone["Advise"]["Thermostat_Write_Tries"]:
                                    self.mutex.acquire()
                                    print("Problem with MPC for zone %s, entering normal schedule." % iter_zone)
                                    print("")
                                    self.mutex.release()
                                    run_mpc_zones[iter_zone] = False
                                    # TODO if here, Add line in LOGGER output that MPC DIDN"T WORK HERE

                    # Log this action if succeeded.
                    if all(succeeded_zones.values()):
                        # TODO If we are in an interval that overlaps with the end of the DR event then
                        # part of the optimization for one interval is in DR and the other outside. We should account
                        # for this.
                        for iter_zone, cfg_zone in cfg_zones.items():
                            # we want to know that we succeeded with all the zones here.
                            did_succeed_zones[iter_zone] = True

                            # Do the log here
                            is_dr = utils.is_DR(now_cfg_timezone, cfg_building)
                            if is_dr:
                                mpc_lambda = cfg_zone["Advise"]["DR_Lambda"]
                            else:
                                mpc_lambda = cfg_zone["Advise"]["General_Lambda"]

                            # TODO update MPCLogger to multizones.
                            MPCLogger.mpc_log(self.building, iter_zone, now_utc, float(cfg_building["Interval_Length"]),
                                              is_mpc=True,
                                              is_schedule=False,
                                              mpc_lambda=mpc_lambda, shut_down_system=False)

                # Running the normal Schedule

                if not all(run_mpc_zones.values()):
                    # Set a dictionary for zones that need to be actuated but have not yet been successful.
                    should_actuate_zones_schedule = {iter_zone: actuate_zones[iter_zone]
                                                                and not did_succeed_zones[iter_zone]
                                                                and not run_mpc_zones[iter_zone]
                                                     for iter_zone in self.zones}

                    normal_schedule_succeeded, action_data = hvac_control(cfg_building, cfg_zones, self.tstats,
                                                                          self.thermal_models,
                                                                          self.zones, self.building, now_utc,
                                                                          consumption_storage=self.consumption_storage,
                                                                          debug=self.debug,
                                                                          actuate_zones=should_actuate_zones_schedule,
                                                                          optimizer=self.normal_schedule, client=self.client)

                    if all(normal_schedule_succeeded.values()):
                        for iter_zone in self.zones:
                            message_data_zones[iter_zone] = action_data[iter_zone]

                            # # Log this action.
                            # if normal_schedule_succeeded:
                            #     MPCLogger.mpc_log(building, self.zone, now_utc, float(cfg_building["Interval_Length"]),
                            #                       is_mpc=False,
                            #                       is_schedule=True,
                            #                       expansion=cfg["Advise"]["Baseline_Dr_Extend_Percent"],
                            #                       shut_down_system=False)
                            # else:
                            #     # If normal schedule fails then we have big problems.
                            #
                            #     # Set the override to false for the thermostat so the local schedules can take over again.
                            #     utils.set_override_false(tstat)
                            #
                            #     # Logging the shutdown.
                            #     MPCLogger.mpc_log(self.building, self.zone, now_utc, float(cfg_building["Interval_Length"]),
                            #                       is_mpc=False, is_schedule=False, shut_down_system=True,
                            #                       system_shut_down_msg="Normal Schedule Failed")
                            #
                            #     print("System shutdown for zone %s, normal schedule has not succeeded. "
                            #           "Returning control to Thermostats. \n" % self.zone)
                            #     break

            # infer actions from setpoints set.
            action_data_zones = {iter_zone: get_action_from_setpoints(cfg_zones[iter_zone],
                                                                      message_data_zones[iter_zone],
                                                                      self.tstats[iter_zone].temperature)
                                 for iter_zone in self.zones}

            print(
                "This process is for building %s" % cfg_building[
                    "Building"])  # TODO Rethink. now every thread will write this.

            # Set the right now times. May involve waiting for a interval number of minutes to pass.
            if not self.simulate:
                # Wait for the next interval if not simulating.
                time.sleep(60. * float(cfg_building["Interval_Length"]) - (
                    (time.time() - starttime) % (60. * float(cfg_building["Interval_Length"]))))

                # set now time.
                now_utc = utils.get_utc_now()

            else:
                # ADVANCE TEMPERATURE OF TSTATS and STORE THE SIMULATION RESULTS

                # NOTE: Making use of the fact that MPCThermalModel stores inside, outside, action data. We
                # will use the action_data from the hvac_control to fill the simulation dictionary as well.

                utc_unix_timestamp = time.mktime(now_utc.timetuple())
                for iter_zone in self.zones:
                    # Get the actions performed for this zone
                    action_zone = action_data_zones[iter_zone]
                    message_data_zone = message_data_zones[iter_zone]

                    self.simulation_results[iter_zone]["inside"][str(utc_unix_timestamp)] = self.tstats[
                        iter_zone].temperature
                    # TODO FIX WEATHER
                    # self.simulation_results[iter_zone]["outside"][str(utc_unix_timestamp)] = \
                    #     self.thermal_models[iter_zone].outside_temperature.iloc[0]
                    self.simulation_results[iter_zone]["cooling_setpoint"][str(utc_unix_timestamp)] = message_data_zone[
                        "cooling_setpoint"]
                    self.simulation_results[iter_zone]["heating_setpoint"][str(utc_unix_timestamp)] = message_data_zone[
                        "heating_setpoint"]
                    self.simulation_results[iter_zone]["state"][str(utc_unix_timestamp)] = action_zone

                    # advancing the temperatures of each zone. each thread is responsible for this.
                    new_temperature, noise = self.tstats[iter_zone].next_temperature(action_zone,
                                                                                     debug=self.debug)
                    if self.debug:
                        self.simulation_results[iter_zone]["noise"][str(utc_unix_timestamp)] = noise

                # waiting for all threads to advance their temperatures before proceeding.
                self.barrier.wait()

                # set now time.
                now_utc += datetime.timedelta(minutes=cfg_building["Interval_Length"])

                # Print the simulation results so far.
                if self.debug:
                    self.mutex.acquire()
                    print("The simulation results for the current thread:")
                    print(self.simulation_results)
                    print("")
                    self.mutex.release()

            # Checking if manual setpoint changes occurred given that we successfully actuated that zone.
            for iter_zone, actuate_zone in actuate_zones.items():
                if actuate_zone and did_succeed_zones[iter_zone]:
                    action_data_zone = action_data_zones[iter_zone]
                    # end program if setpoints have been changed. (If not writing to tstat we don't want this)
                    if action_data_zone is not None and utils.has_setpoint_changed(self.tstats[iter_zone],
                                                                                   action_data_zone,
                                                                                   iter_zone, self.building):
                        # Tell the logger to record this setpoint change.
                        MPCLogger.mpc_log(self.building, iter_zone, utils.get_utc_now(),
                                          float(cfg_building["Interval_Length"]),
                                          is_mpc=False, is_schedule=False, shut_down_system=True,
                                          system_shut_down_msg="Manual Setpoint Change")
                        print("Ending program for zone %s due to manual setpoint changes. \n" % iter_zone)
                        # Do not actuate the zone again till the end of the program
                        stop_program_zone[iter_zone] = True


        # if actuate:
        #     # TODO FIX THE UPDATE STEP.
        #     # NOTE: call update before setWeatherPredictions and set_temperatures
        #     thermal_model.update(zone_temperatures, interval=cfg["Interval_Length"])

        if not self.simulate:
            # Set the override to false for the thermostat so the local schedules can take over again.
            for iter_zone in self.zones:
                utils.set_override_false(self.tstats[iter_zone])

        # making sure that the barrier does not account for this zone anymore.
        self.mutex.acquire()
        self.barrier.num_threads -= 1
        self.mutex.release()


def main(building, optimization_type, simulate=False, debug=True, run_server=True, start_simulation=None, end_simulation=None):
    """
    Intializes the control functions and starts them.
    :param building: (string) Building name 
    :param optimization_type: (string) ["LP", "DP"] Type of optimization to use. 
                    "LP" is recommended for a fast solver, "DP" is recommended for accuracy.
    :param simulation: (Boolean) Whether to simulate or run standard control. Will not actuate tstats if set to True and 
            will use star_simulat and end_simulation for the simulation range.
    :param debug: (bool) Whether to print debug messages. Will also not actuate the tstats if set to true.
    :param run_server: (bool) for whethe to run on server. Will use local get_client() method if not on server, 
                    otherwise config file needs to specify the location of xbos entity. 
    :param start_simulation: (datetime.datetime timezone aware) The start of time range for which to run the simulation. 
                                Needs to be not None is simulation is True.
    :param end_simulation: (datetime.datetime timezone aware) The end of time range for which to run the simulation. 
                                Needs to be not None is simulation is True.
    :return: if simulate is true it will return a dictionary of simultion results. Otherwise, return None. 
    """
    if simulate:
        assert start_simulation is not None and end_simulation is not None

    # Set building fields.
    zones = utils.get_zones(building)
    cfg_building = utils.get_config(building)

    # Get the client.
    # It will try to use the Server details given in the config file if run_server is true.
    # Otherwise it will run the local xbos version.
    if run_server:
        client = utils.choose_client(cfg_building)
    else:
        client = utils.choose_client()

    # Get hod client.
    hod_client = HodClient("xbos/hod", client)

    # Getting the tstats for the building.
    tstats_building = utils.get_thermostats(client, hod_client, cfg_building["Building"])

    # --- Thermal Model Init ------------
    # initialize and fit thermal model
    thermal_data = utils.get_data(cfg=cfg_building, client=client, days_back=150, force_reload=False)

    zone_thermal_models = {}
    for iter_zone, zone_data in thermal_data.items():
        # Concat zone data to put all data together and filter such that all datapoints have dt != 1
        filtered_zone_data = zone_data[zone_data["dt"] == 5]
        if iter_zone != "HVAC_Zone_Please_Delete_Me":
            zone_thermal_models[iter_zone] = MPCThermalModel(zone=iter_zone, thermal_data=filtered_zone_data,
                                                        interval_length=15, thermal_precision=0.05)
    if debug:
        print("Trained Thermal Model")
    # --------------------------------------

    # setting the thermostats to be used for control.
    if simulate:
        # Set thermostats for simulation.
        tstats_simulation = {}
        curr_temperatures_simulation = {zone: dict_tstat.temperature for zone, dict_tstat in tstats_building.items()}

        for iter_zone, thermal_model_zone in zone_thermal_models.items():
            zone_simulate_tstat = SimulationTstat(thermal_model_zone, curr_temperatures_simulation[iter_zone])
            zone_simulate_tstat.set_gaussian_distributions(thermal_data[iter_zone], thermal_data[iter_zone]["t_next"])
            tstats_simulation[iter_zone] = zone_simulate_tstat

        tstats_control = tstats_simulation
    else:
        tstats_control = tstats_building

    # Get all the optimizers and normal schedule
    dp_optimizer = IteratedDP(building, client, len(tstats_control.keys()), num_iterations=1)
    lp_optimizer = AdviseLinearProgram(building, client, num_threads=1, debug=False)
    normal_schedule = NormalSchedule(building, client, num_threads=len(tstats_control.keys()), debug=None)

    # Set consumption storage
    consumption_storage = None

    # initialize the control
    if optimization_type == "DP":
        # setting a barrier for the threads. this will be used in hvac_control for stopping all threads before setting
        # all zone_temperatures and weather for the thermalModel
        num_threads = len(tstats_control.keys())
        thread_barrier = utils.Barrier(num_threads)
        thread_mutex = threading.Semaphore(1)
        threads = []
        for zone in zones:
            # TODO only because we want to only run on the basketball courts.
            if building != "jesse-turner-center" or "Basketball" in zone:
                thread = ZoneThread(tstats_control, [zone], client, zone_thermal_models[zone],
                                    cfg_building["Building"], thread_barrier, thread_mutex,
                                    consumption_storage=consumption_storage, normal_schedule=normal_schedule,
                                    optimization_type=optimization_type, dp_optimizer=dp_optimizer,
                                    lp_optimizer=lp_optimizer,
                                    debug=debug, simulate=simulate,
                                    simulate_start=start_simulation, simulate_end=end_simulation)
                thread.start()
                threads.append(thread)
    elif optimization_type == "LP":
        # setting a barrier for the threads. this will be used in hvac_control for stopping all threads before setting
        # all zone_temperatures and weather for the thermalModel
        thread_barrier = utils.Barrier(1)
        thread_mutex = threading.Semaphore(1)
        threads = []
        thread = ZoneThread(tstats_control, zones, client, zone_thermal_models,
                            cfg_building["Building"], thread_barrier, thread_mutex,
                            consumption_storage=consumption_storage, normal_schedule=normal_schedule,
                            optimization_type=optimization_type, dp_optimizer=dp_optimizer,
                            lp_optimizer=lp_optimizer,
                            debug=debug, simulate=simulate,
                            simulate_start=start_simulation, simulate_end=end_simulation)
        thread.start()
        threads.append(thread)
    else:
        raise Exception("ERROR: Invalid Optimization type given.")

    for t in threads:
        t.join()

    if debug:
        for t in threads:
            print(t.simulation_results)

    if simulate:
        simulation_results = {}
        for t in threads:
            simulation_results[tuple(t.zones)] = t.simulation_results


if __name__ == '__main__':
    # building, optimization_type, simulate=False, debug=True, run_server=True, start_simulation=None, end_simulation=None

    import os
    def ask_input(ask_name, option_list):
        print "-----------------------------------"
        print ask_name + ":"
        print "-----------------------------------"
        for idx, option in enumerate(option_list, start=1):
            print idx, option
        print "-----------------------------------"
        idx = input("Please choose a building (give a number):") - 1
        chosen_option = option_list[idx]
        print "-----------------------------------"
        print "Option chosen: " + str(chosen_option)
        print "-----------------------------------"
        return chosen_option

    def ask_date(date_type):
        print "-----------------------------------"
        print  "Input all the dates for %s and time in the timezone of the building." % date_type
        print "-----------------------------------"
        day_chosen = int(input("Day: "))
        assert 1 <= day_chosen <= 31
        month_chosen = int(input("Month: "))
        assert 1 <= month_chosen <= 12
        year_chosen = int(input("Year: "))
        assert 1900 <= year_chosen <= utils.get_utc_now().year + 1
        hour_chosen = int(input("Hour: "))
        assert 0 <= hour_chosen <= 24
        minute_chosen = int(input("Minute: "))
        assert 0 <= minute_chosen <= 60

        date_datetime = datetime.datetime(year=year_chosen, month=month_chosen, day=day_chosen, hour=hour_chosen, minute=minute_chosen)

        print "-----------------------------------"
        print "Date and time chosen: " + utils.get_datetime_to_string(date_datetime)
        print "-----------------------------------"
        return datetime.datetime(year=year_chosen, month=month_chosen, day=day_chosen, hour=hour_chosen, minute=minute_chosen)

    def ask_end_date(start_date):
        print "-----------------------------------"
        hours_after_start = int(input("How many hours after the start of simulation %s should the end be: " % utils.get_datetime_to_string(start_date)))
        end_date = start_date + datetime.timedelta(hours=hours_after_start)
        print "-----------------------------------"
        print "Date and time chosen:" + utils.get_datetime_to_string(end_date)
        print "-----------------------------------"
        return end_date

    building = ask_input("Building", os.walk(utils.SERVER_DIR_PATH + "/Buildings/").next()[1])
    optimization_type = ask_input("Optimization Type", ["DP", "LP"])
    simulate_string = ask_input("Should simulate", ["Yes", "No"])
    simulate = simulate_string == "Yes"
    debug_string =  ask_input("Should debug", ["Yes", "No"])
    debug = debug_string == "Yes"
    run_server_string =  ask_input("Should run on server", ["Yes", "No"])
    run_server = run_server_string == "Yes"
    if simulate:
        start_simulation = ask_date("start")
        end_simulation = ask_end_date(start_simulation)
    else:
        start_simulation = None
        end_simulation = None


    result = main(building, optimization_type, simulate, debug, run_server, start_simulation, end_simulation)



