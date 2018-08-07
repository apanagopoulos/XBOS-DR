import datetime

# Linear program solver.
import gurobipy
import numpy as np
import threading

import utils

class AdviseLinearProgram:
    def __init__(self, building, num_threads=None, debug=False):
        self.building = building
        self.zones = utils.get_zones(building)
        self.debug = debug

        if num_threads is None:
            num_threads = len(self.zones)
        # Initialize the barrier to wait for all zones to get together
        self.barrier = utils.Barrier(num_threads)
        # Get mutex to block other threads from running
        self.mutex = threading.Semaphore(1)

        # initalize all temperatures variable for the linear program. Will be reset after every run of linear program.
        self.all_zone_temperatures = {}
        # helper variable which will be used in start_linear_program function. If the run_linear_program has happened
        # it will be set to True, and reset to False only for most of the duration of the run_linear_program
        self.ran_linear_program = False

        # shared variable to get the data of the last linear_program_optimization
        self.last_linear_program_optimization_data = None

    def run_linear_program(self, start, end, zones, zone_starting_temperatures):
        # Actions to perform berfore linear program optimization
        self.mutex.acquire()
        self.ran_linear_program = False
        # all zones will now populate the all_temperatures variable at once.
        for zone in zones:
            self.all_zone_temperatures[zone] = zone_starting_temperatures[zone]
        self.mutex.release()

        self.barrier.wait()



        # Only one zone will get to run the linear program
        self.mutex.acquire()
        if not self.ran_linear_program:
            linear_program = LinearProgram(building=self.building, start=start, end=end,
                                           starting_zone_temperatures= self.all_zone_temperatures,
                                           debug=self.debug)
            # TODO following functions should be called in optimize i guess?
            linear_program.set_inside_temperature()
            linear_program.set_discomfort()
            linear_program.set_objective_function()
            linear_program.lp_solver.optimize()
            # TODO set shared linear program variable accordingly.
            for v in lp.lp_solver.getVars():
                print(v.varName, v.x)
            self.ran_linear_program = True
        self.mutex.release()








class LinearProgram:
    """The linear porgram class."""

    def __init__(self, building, start, end, starting_zone_temperatures, debug=False):
        """
        
        :param building: (str) building name
        :param start: datetime timezone aware
        :param end: datetime timezone aware
        :param starting_zone_temperatures: {str zone: float temperature}
        """
        self.debug = debug

        # set control variables.
        self.building = building
        self.zones = utils.get_zones(building)
        self.building_cfg = utils.get_config(building)
        self.zone_advise = {zone: utils.get_zone_config(building, zone) for zone in self.zones}
        self.interval = float(self.building_cfg["Interval_Length"])  # in minutes
        self.start = start
        self.end = end
        self.num_timesteps = int((self.end - self.start).seconds // (self.interval * 60))

        # TODO think if all times are properly synchronized. Especially with the num timesteps having too many too little steps.
        # TODO The util functions have end inclusive, but that gives too many intervals.
        # set all data matrices besides thermal model.
        # All are {zone: pd.df with respective columns but same timeseries}
        self.lambda_cost_discomfort = utils.get_lambda_matrix(self.building, self.start, self.end, self.interval)
        self.occupancy = utils.get_occupancy_matrix(self.building, self.start, self.end, self.interval)
        self.comfortband = utils.get_comfortband_matrix(self.building, self.start, self.end, self.interval)
        # TODO add Safety contraints to the linear program. Simply add them in the temperature variables for LP.
        self.prices = utils.get_price_matrix(self.building, self.start, self.end, self.interval)
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
                                                                   name="cool_zone{" + str(zone) + "}_time{" + str(
                                                                       time) + "}") for
                                             time in range(self.num_timesteps - 1)])
                               for zone in self.zones}

        self.action_heating = {zone:
                                   np.array([self.lp_solver.addVar(lb=0, ub=1.0,
                                                                   name="heat_zone{" + str(zone) + "}_time{" + str(
                                                                       time) + "}") for
                                             time in range(self.num_timesteps - 1)])
                               for zone in self.zones}

        # The inside temperatures
        self.t_in = {zone: np.array([self.lp_solver.addVar(name="temperature_zone{" +
                                                                str(zone) + "}_time{" + str(time) + "}") for
                                     time in range(self.num_timesteps)])
                     for zone in self.zones}
        # set starting temperatures
        for zone in self.zones:
            self.t_in[zone][0] = starting_zone_temperatures[zone]

        # The thermal consumption for each zone
        self.thermal_consumption = None

        # The discomfort
        self.discomfort = {zone: np.array([self.lp_solver.addVar(name="discomfort_zone{" +
                                                                      str(zone) + "}_time{" + str(time) + "}") for
                                           time in range(self.num_timesteps)])
                           for zone in self.zones}

        # The total cost
        self.total_cost = None

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
    BUILDING = "ciee"
    ZONES = utilsLinear.get_zones(BUILDING)
    start = utilsLinear.get_utc_now()
    end = start + datetime.timedelta(hours=4)
    lp = LinearProgram(BUILDING, start, end, {zone: 70 for zone in ZONES}, debug=True)
    lp.set_inside_temperature()
    lp.set_discomfort()
    lp.set_objective_function()
    lp.lp_solver.optimize()
    for v in lp.lp_solver.getVars():
        print(v.varName, v.x)
