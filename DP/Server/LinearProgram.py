import datetime
import json
import threading
from collections import defaultdict
import time
import math

# Linear program solver.
import gurobipy
import numpy as np
import pytz

import utils
from ThermalDataManager import ThermalDataManager


def to_var(var_type, zone, time_step):
    """
    Helper function to encode information to variable string.
    :param var_type: (str) i.e. "t_in", "cool", etc...
    :param zone: (str) 
    :param time_step: (int) 
    :return: (str) encoded information which from_var function can decode 
    """
    var_dict = {"var_type": var_type, "zone": zone, "time_step": int(time_step)}
    return json.dumps(var_dict)


def from_var(var_str):
    """
    Helper function to decode information from variable string.
    :param var_str: (str) as returned by to_var
    :return: dictionary with {"var_type": var_type, "zone": zone, "time_step": time_step}
    """
    return json.loads(var_str)


class AdviseLinearProgram:
    def __init__(self, building, client, num_threads=1, debug=False):
        self.building = building
        # self.zones = utils.get_zones(building)
        self.debug = debug
        self.client = client

        # set threads
        self.num_threads = num_threads

        # shared variable to get the data of the last linear_program_optimization
        self.last_lp_data = defaultdict(lambda: defaultdict(dict))

        # set the datamangers for all possible zones for the given building
        self.data_managers_zones = utils.get_zone_data_managers(building, utils.get_zones(building), None, client)
        self.thermal_data_manager = ThermalDataManager(utils.get_config(building), client, interval=5)

    def advise(self, start, end, zones, cfg_building, cfg_zones, thermal_models,
               zone_starting_temperatures, consumption_storage):

        # reset the last_lp_data
        self.last_lp_data = defaultdict(lambda: defaultdict(dict))

        # set start and end times.
        start_utc = start.astimezone(tz=pytz.utc)
        end_utc = end.astimezone(tz=pytz.utc)

        start_cfg_timezone = start_utc.astimezone(tz=utils.get_config_timezone(cfg_building))
        end_cfg_timezone = end_utc.astimezone(tz=utils.get_config_timezone(cfg_building))

        # For thermal model need to set weather predictions for every loop and set current zone temperatures.
        outside_temperatures = utils.get_outside_temperatures(cfg_building, start_utc, end_utc,
                                                              self.data_managers_zones.values()[0],
                                                              self.thermal_data_manager, interval=15)
        for zone, zone_thermal_model in thermal_models.items():
            zone_thermal_model.set_temperatures(zone_starting_temperatures)
            zone_thermal_model.set_outside_temperature(outside_temperatures)

        # get datamangers to use
        data_managers_to_use = {}
        for iter_zone, data_manager in self.data_managers_zones.items():
            if iter_zone in zones:
                data_managers_to_use[iter_zone] = data_manager

        # Initialize Linear Program
        linear_program = LinearProgram(building=self.building, zones=zones, start=start_cfg_timezone,
                                       end=end_cfg_timezone,
                                       starting_zone_temperatures=zone_starting_temperatures,
                                       data_managers=data_managers_to_use, thermal_models=thermal_models,
                                       debug=self.debug)
        # TODO following functions should be called in optimize i guess?
        # set linear Program parameters
        linear_program.set_inside_temperature()
        linear_program.set_discomfort()
        linear_program.set_objective_function()

        # Run optimization
        linear_program.lp_solver.optimize()

        # set the results into a data dictionary
        for v in linear_program.lp_solver.getVars():
            var_dict = from_var(v.varName)
            var_type, var_zone, var_time = var_dict["var_type"], var_dict["zone"], var_dict["time_step"]
            var_value = v.x
            # set the variable information in our data container.
            self.last_lp_data[var_zone][var_time][var_type] = var_value

        if self.debug:
            print("The lp data: %s" % str(self.last_lp_data))

        # get the first action to return for control
        advise_data = {}
        for zone in zones:
            heating_action = self.last_lp_data[zone][0]["heat"]
            cooling_action = self.last_lp_data[zone][0]["cool"]

            if heating_action != 0 and cooling_action != 0:
                raise Exception("ERROR: In the linear program optimization for zone %s one of the actions was not"
                                "0. This may indicate no or negative energy prices or a bug in the LP. "
                                "Heating: %f, Cooling: %f." % (zone, heating_action, cooling_action))

            if heating_action > cooling_action:
                advise_data[zone] = utils.HEATING_ACTION
            elif cooling_action > heating_action:
                advise_data[zone] = utils.COOLING_ACTION
            else:
                advise_data[zone] = utils.NO_ACTION

        return advise_data, None


class LinearProgram:
    """The linear porgram class."""
    # NOTE: Assuming thermal_interval % interval = 0.

    def __init__(self, building, zones, start, end, starting_zone_temperatures, data_managers, thermal_models, debug=False):
        """
        
        :param building: (str) building name
        :param start: datetime timezone aware
        :param end: datetime timezone aware
        :param data_managers: {zone: datamanager} To use for datamethods. 
        :param starting_zone_temperatures: {str zone: float temperature}
        """
        self.debug = debug

        # set control variables.
        self.building = building
        self.zones = zones
        self.building_cfg = utils.get_config(building)
        self.zone_advise = {zone: utils.get_zone_config(building, zone) for zone in self.zones}
        self.interval = float(self.building_cfg["Interval_Length"])  # in minutes
        self.start = start
        self.end = end
        self.num_timesteps = int((self.end - self.start).seconds // (self.interval * 60))

        # set datamanagers
        self.datamanager_zones = data_managers

        # set thermal models
        self.thermal_models = thermal_models

        # TODO think if all times are properly synchronized. Especially with the num timesteps having too many too little steps.
        # TODO The util functions have end inclusive, but that gives too many intervals.
        # set all data matrices besides thermal model.
        # All are {zone: pd.df with respective columns but same timeseries}
        self.lambda_cost_discomfort = utils.get_lambda_matrix(self.building, self.zones, self.start, self.end,
                                                              self.interval, self.datamanager_zones)
        self.occupancy = utils.get_occupancy_matrix(self.building, self.zones, self.start, self.end, self.interval,
                                                    self.datamanager_zones)
        self.comfortband = utils.get_comfortband_matrix(self.building, self.zones, self.start, self.end, self.interval,
                                                        self.datamanager_zones)
        # TODO add Safety contraints to the linear program. Simply add them in the temperature variables for LP.
        self.prices = utils.get_price_matrix(self.building, self.zones, self.start, self.end, self.interval,
                                             self.datamanager_zones)
        # TODO Add outside temperatures. For now we will work with a constant thermal model.

        # Two stage should be somewhat easy to extend to by allowing a higher power. Depends on 2 stage logic though.
        # Using self.interval/60. to adjust consumption to dollar/kwInterval
        self.power_heating = {zone: self.zone_advise[zone]["Advise"]["Heating_Consumption"] * self.interval / 60. for
                              zone in self.zones}
        self.power_cooling = {zone: self.zone_advise[zone]["Advise"]["Cooling_Consumption"] * self.interval / 60. for
                              zone in self.zones}

        # set up the linear program solver from gurobipy
        self.lp_solver = gurobipy.Model("solver")

        # Setting up the variables for the Linear Program
        # TODO there has to be timesteps-1 actions because action is an edge in the tree.
        # The thermal actions to take. Either heating or cooling.
        # These two variables are the ones which we will in the end want to know
        # The variables range from 0 to 1 and represent the percentage of the heating/cooling power that should be used.
        # NOTE, since both variables incur cost when above 0, it should never happen that both action_cooling
        # and action_heating have variables at the same time and zone which are above zero, because this would not be
        # optimal. e.g. If the linear program wants to achieve a temperature increase of one degree, it could do
        # so by heating a lot and by cooling it down at the same time, or by just heating a bit less. The latter would
        # be the optimal path.
        # If there is ever a time where both are above zero, then the LP should be evaluated since
        # that should be an indicator for a bug.
        self.action_cooling = {zone:
                                   np.array([self.lp_solver.addVar(lb=0, ub=1.0,
                                                                   name=to_var("cool", zone, time)) for
                                             time in range(self.num_timesteps - 1)])
                               for zone in self.zones}

        self.action_heating = {zone:
                                   np.array([self.lp_solver.addVar(lb=0, ub=1.0,
                                                                   name=to_var("heat", zone, time)) for
                                             time in range(self.num_timesteps - 1)])
                               for zone in self.zones}

        # make sure that interval % thermal_interval = 0.
        assert all([self.interval % self.thermal_models[iter_zone].interval_thermal == 0 for iter_zone in self.zones])

        # The inside temperatures. Since the thermal model may learn predictions which are not for same interval length
        # as we want.
        self.t_in = {zone: np.array([self.lp_solver.addVar(name=to_var("t_in", zone, time)) for
                                     time in range(
                int(self.num_timesteps * math.ceil(self.interval/self.thermal_models[zone].interval_thermal)))])
                     for zone in self.zones}

        # set starting temperatures
        for zone in self.zones:
            self.t_in[zone][0] = starting_zone_temperatures[zone]

        # The thermal consumption for each zone
        self.thermal_consumption = None

        # The discomfort
        self.discomfort = {zone: np.array([self.lp_solver.addVar(name=to_var("discomfort", zone, time)) for
                                           time in range(self.num_timesteps)])
                           for zone in self.zones}

        # The total cost
        self.total_cost = None

    def _set_inside_temperature(self):
        """
        Sets the inside temperature constraints for each zone.
        NOTE: Now just using a constant model. If heat we add 0.1 degrees and when cool we subtract.
        :return: None
        """
        # set the data matrix for thermal model to create the features
        for iter_zone in self.zones:
            thermal_model = self.thermal_models[iter_zone]
            thermal_interval = thermal_model.interval_thermal
            for time in range(1, len(self.t_in)):

                # getting the action for last interval. because that is when we start and advance forward.
                if ((time - 1) * thermal_interval % self.interval) == 0:
                    action_cooling = self.action_cooling[iter_zone][(time - 1) / int(self.interval / thermal_interval)]
                    action_heating = self.action_heating[iter_zone][(time - 1) / int(self.interval / thermal_interval)]

                    # The last outside temperatures to get the current inside temperature.
                    t_out = thermal_model.outside_temperature.loc[
                        self.start + datetime.timedelta(minutes=(time - 1) / int(self.interval / thermal_interval))]

                # get relevant temperatures
                prev_temperature = self.t_in[iter_zone][time-1]
                curr_temperature = self.t_in[iter_zone][time]
                other_zone_temperatures = thermal_model.zone_temperatures


                # to get the zone order in which to apply the coefficients
                # ["heating", 'cooling','two_stage_heating', 'two_stage_cooling','t_out', 'bias'] + list(zone_col)
                param_oder = thermal_model.thermal_model._params_coeff_order
                zone_list = param_oder[6:]

                # Get coefficients and biases
                coeffs = thermal_model.linear_thermal_model._params[:len(param_oder)]
                biases = thermal_model.linear_thermal_model._params[len(param_oder):]

                # other zone contribution
                other_zone_contribution = 0
                for i in range(len(zone_list)):
                    other_zone_contribution = (prev_temperature - other_zone_temperatures[zone_list[i][len("zone_temperature_"):]]
                                               - biases[i + 2]) * coeffs[i+6]

                # set up equality. We will skip two stage for now.
                self.lp_solver.addConstr(curr_temperature == prev_temperature +
                                         action_heating * coeffs[0] +
                                         action_cooling * coeffs[1] +
                                         (prev_temperature - t_out - biases[0]) * coeffs[4] +
                                         (-biases[1])* coeffs[5]+
                                         other_zone_contribution)

    def set_inside_temperature(self):
        """
        Sets the inside temperature constraints for each zone.
        NOTE: Now just using a constant model. If heat we add 0.1 degrees and when cool we subtract.
        :return: None
        """
        HEAT = 0.1
        COOL = -0.1
        # Using the timesteps as outer loop for when having to link the zone temperatures.
        # We already know 0th time, so no need to set it.
        for time in range(1, self.num_timesteps):
            for zone in self.zones:
                last_temperature = self.t_in[zone][time - 1]
                # getting the action for last interval. because that is when we start and advance forward.
                action_cooling = self.action_cooling[zone][time - 1]
                action_heating = self.action_heating[zone][time - 1]

                self.lp_solver.addConstr(self.t_in[zone][time] == last_temperature +
                                         action_cooling * COOL + action_heating * HEAT)


    def set_discomfort(self):
        """Sets the discomfort constraints. Discomfort is the distance to the comfortband.
        :return: None."""
        for zone in self.zones:
            for time in range(self.num_timesteps):
                curr_temperature = self.t_in[zone][time]
                curr_discomfort = self.discomfort[zone][time]
                comfortband = self.comfortband[zone].iloc[
                    time]  # TODO check if works as intended. MAYBE print the time or something
                t_high = comfortband["t_high"]
                t_low = comfortband["t_low"]

                # TODO maybe not need to use time=0 because we know what it should be. We could actually debug with it.
                # draw the real number line and evaluate the three cases to visualize why this works.
                # Three cases are where temperature is in comfortband, is over comfortband and under comfortband.
                self.lp_solver.addConstr(curr_discomfort >= curr_temperature - t_high)
                self.lp_solver.addConstr(curr_discomfort >= t_low - curr_temperature)
                self.lp_solver.addConstr(curr_discomfort >= 0)

    def set_consumption(self):
        """Sets the consumption """
        pass

    def set_objective_function(self):
        if self.debug:
            print("Setting Objective.")
        objective = gurobipy.LinExpr()
        # https: // www.gurobi.com / documentation / 8.0 / refman / cpp_oper_times.html

        # add the discomfort
        for zone in self.zones:
            curr_lambda = self.lambda_cost_discomfort[zone].iloc[:self.num_timesteps].values.reshape(self.num_timesteps)
            objective += self.discomfort[zone].dot(curr_lambda)

        # add cost
        for zone in self.zones:
            # TODO Note that this is because actions are edges. and therefore consumption is only for edges.
            # TODO Discomfort is for nodes.
            curr_prices = self.prices[zone].iloc[:self.num_timesteps - 1].values.reshape(self.num_timesteps - 1)
            curr_lambda = self.lambda_cost_discomfort[zone].iloc[:self.num_timesteps - 1].values.reshape(
                self.num_timesteps - 1)

            cost_heating = self.action_heating[zone] * self.power_heating[zone] * curr_prices
            cost_cooling = self.action_cooling[zone] * self.power_cooling[zone] * curr_prices
            objective += (cost_heating + cost_cooling).dot(curr_lambda)

        self.lp_solver.setObjective(objective, gurobipy.GRB.MINIMIZE)


if __name__ == "__main__":
    import time

    s_time = time.time()
    BUILDING = "ciee"
    ZONES = utils.get_zones(BUILDING)
    start = utils.get_utc_now()
    end = start + datetime.timedelta(hours=4)
    lp = LinearProgram(BUILDING, ZONES, start, end, {zone: 70 for zone in ZONES}, debug=True)
    lp.set_inside_temperature()
    lp.set_discomfort()
    lp.set_objective_function()
    lp.lp_solver.optimize()
    for v in lp.lp_solver.getVars():
        print(v.varName, v.x)

    print(time.time() - s_time)
