import numpy as np
from gurobipy import *



class LinearZones:
    def __init__(self, price_buy, price_sell, max_power_heating, max_power_cooling, num_zones, num_time_steps, total_time,
                 peak_cost=8.03):
        """
        :param price_buy: A function of time and HVAC-Zone which returns the cost to buy a kilowatt at the given time and 
        for the given zone. (Matrix zone x time)
        :param price_sell: A function of time and HVAC-Zone which returns the revenue of selling a kilowatt at the given time and 
        for the given zone. (Matrix zone x time)
        :param max_power_heating: A matrix (zone x time) indicating the maxPower to be used for heating in any zone at any time.
        :param max_power_cooling: A matrix (zone x time) indicating the maxPower to be used for cooling in any zone at any time.
        :param num_zones: The number of HVAC zones.
        :param num_time_steps: The number of times steps from now to Horizon.
        :param total_time: The total time in minutes of the timeframe.
        :param peak_cost: The cost at which the peak consumption is priced. 
        """
        self.price_buy = price_buy  # look at DP EnergyConsumption.py
        self.max_power_heating = max_power_heating  # TODO Marco. And we are not having same power for cooling and heating.
        self.max_power_cooling = max_power_cooling  # TODO Note changed logic in script for this. maxCooling when action is negative and vice versa.
        self.num_zones = num_zones
        self.num_time_steps = num_time_steps
        self.peakCost = peak_cost  # peack_cost dollars/kW

        # The last peak demands
        self.last_peak_demand = 0  # TODO needs to be filled in.

        # self.leakage_rate = np.ones(
        #     self.num_zones)  # TODO Thermal model implement as in paper "AEC of Electricity-base Space Heating Systems" for each zone.

        # To calculate energy from power.
        self.delta_time = float(total_time) / self.num_time_steps

        # This is the heating efficiency parameter. Should be an array where the index is the corresponding zone.
        # self.heating_efficiency = np.ones(self.num_zones)  # For now it doesn't do anything.

        # Will represent the probability of occupation.
        # is intended to return a matrix where the row represents the ith zone and column the jth time-step.
        self.occupationModel = Occupation()  # TODO

        # will be used to get the outside temperature.
        # is intended to return a matrix where the row represents the ith zone and column the jth time-step.
        self.t_out = np.zeros(self.num_zones, self.num_time_steps)  # "TODO. Ask Gabe."

        # will be used to get the temperature setpoints.
        # A matrix (num_zones x num_time_steps)
        self.temperatureSetpoint = 0  # TODO load from config. Or just set with Discomfort. set two variables.

        # The prediction of the power consumption of the building as a num_zones x num_time_steps matrix.
        self.powerModel = np.zeros(self.num_zones,
                                   self.num_time_steps)  # TODO work later. for now set to zero to only evaluate heating."

        # TODO Look into the weighing of each objective term.
        # discounting variable.
        # array num_time_steps.
        self.discounting = np.ones(self.num_time_steps)  # TODO Check if right. But leave as one.

        # comfort and cost balancing
        self.comfortBalancing = np.ones(self.num_zones) - 0.1111  # TODO Implement for cost (1-\lambda). Think about it

        # the linear programming solver from gurobipy.
        self.model = Model("solver")

        # setting the thermal model. Matrix (zones x timeSteps)
        self.t_in = np.array([[
            self.model.addVar("Temperature_in_zone{" + str(z) + "}_time{" + str(t) + "}") for t in
            range(self.num_time_steps)
        ] for z in range(self.num_zones)])

        # We changed the following commented out to have only self.action.
        '''
        the variables for the linear program. They control heat or cool. For now we will use binary variables.
        # has dimension numZone x num_time_steps.
        self.heat = np.array([[self.model.addVar(
            vtype=GRB.BINARY, name="heat_zone{" + str(zone) + "}_time{" + str(time) + "}") for time in
            range(self.num_time_steps)]
            for zone in range(self.num_zones)])

        self.cool = np.array([[self.model.addVar(
            vtype=GRB.BINARY, name="cool_zone{" + str(zone) + "}_time{" + str(time) + "}") for time in
            range(self.num_time_steps)]
            for zone in range(self.num_zones)])
        '''

        # TODO Because cooling_power need not equal heating_power, just adjust 1 to 0.8 if heating power is
        # 0.8 of cooling_power.

        # This will be the eventual action to be taken.
        # We constraint -1 <= self.action <= 1 and self.action is continuous.
        # Where negative implies that we use that percentage of max_power_cooling and positive
        # for percentage of max_power_heating.
        self.action = np.array([[self.model.addVar(lb=-1.0, ub=1.0,
                                                   name="cool_zone{" + str(zone) + "}_time{" + str(time) + "}") for time
                                 in
                                 range(self.num_time_steps)]
                                for zone in range(self.num_zones)])

        # Discomfort array of the linear program where index is the zone. Used for objective.
        self.discomfort = np.array([self.model.addVar(
            name="discomfort_zone{" + str(zone) + "}") for zone in range(self.num_zones)])

        # Heating Consumption matrix of LP without powerModel. num_zones x num_time_steps.
        self.heatingConsumption = np.array([
            [
                self.model.addVar(
                    name="heating_consumption_zone{" + str(zone) + "}_time{" + str(t) + "}")
                for t in range(self.num_time_steps)]
            for zone in range(self.num_zones)])

        # Right now as total energy consumptions. One entry if for consumption if sold and the other for
        # if bought
        self.totalConsumption = np.array([
            [
                [self.model.addVar(
                    name="consumption_sell_zone{" + str(zone) + "}_time{" + str(t) + "}"),
                    self.model.addVar(
                        name="consumption_buy_zone{" + str(zone) + "}_time{" + str(t) + "}")
                ]
                for t in range(self.num_time_steps)]
            for zone in range(self.num_zones)])

        # The peak demand charge for the given time period.
        self.peakCharge = self.model.addVar(name="peakDemandCharge")

    def constrain_action(self):
        """Constraints the actions, in case I make them non Binary. TODO Upper and lower bounds can actually be set in 
        the addVar step"""
        for z in range(self.num_zones):
            for t in range(self.num_time_steps):
                self.model.addConstr(self.cool[z][t] >= 0)
                self.model.addConstr(self.cool[z][t] <= 1)

                self.model.addConstr(self.heat[z][t] >= 0)
                self.model.addConstr(self.heat[z][t] <= 1)

                self.model.addConstr(self.heat[z][t] - self.cool[z][t] >= -1)
                self.model.addConstr(self.heat[z][t] - self.cool[z][t] <= 1)

    def set_objective(self):
        obj = LinExpr()
        obj += self.discomfort.dot(self.comfortBalancing)  # add the discomfort term
        obj += self.totalConsumption
        temp_rev = self.totalConsumption[:, 0] * self.price_sell  # setting the revenue we might make.
        temp_cost = self.totalConsumption[:, 1] * self.price_buy  # setting the cost we might have to pay.
        obj += temp_rev.dot(self.discounting) + temp_cost.dot(self.discounting)  # cost term added
        obj += self.peakCharge / self.delta_time * self.peakCost * 1 / self.num_time_steps  # add demand charge term from paper "Smart Charge"

        self.model.setObjective(obj, GRB.MAXIMIZE)

    # TODO Change because we have comfortband.
    def set_discomfort(self):
        """Set the constraints for the discomfort. We are setting the total discomfort of a zone from now till the
        end of the time horizon, since the objective in the paper has a double sum which can be interchanged."""
        TDiff = (
                    self.temperatureIn - self.temperatureSetpoint) * self.discounting  # TODO because TDiff is a varialbe. use discounting variable here for discomfort
        for zone in range(self.num_zones):
            # have to set two constraints to work with abs() values. We would actually like to work with the
            # absolute value temperature difference, which is not allowed in LP's. So, we use a trick
            # where we want the variable to be greater or equal to both the difference and negated difference.
            # A picture of the real number line might help to see why this trick works.
            self.model.addConstr(self.discomfort[zone] >= self.occupationModel[zone].dot(
                TDiff[zone]))  # TODO fix the occupation model and how they are multiplied
            self.model.addConstr(
                self.discomfort[zone] >= -self.occupationModel[zone].dot(
                    TDiff[zone]))  # TODO fix the occupation model and how they are multiplied

    def set_heating_consumption(self):
        """Set the consumption for heating/cooling for every zone at every time step. Needs to be done because
        we can have negative valued actions."""
        for z in range(self.num_zones):
            for t in range(self.num_time_steps):
                # using a trick where when the action is negative the self.action * self.max_power_cooling larger and
                # vice versa.
                self.model.addConstr(
                    self.heatingConsumption[z][t] >= -self.action[z][t] * self.max_power_cooling[z][t] * self.delta_time)
                self.model.addConstr(
                    self.heatingConsumption[z][t] >= self.action[z][t] * self.max_power_heating[z][t] * self.delta_time)

    def set_total_consumption(self):
        """Set consumption for the whole building on time-step level. Sets energy quantities."""
        Cons_C = [sum(self.heatingConsumption[:, t]) for t in self.num_time_steps]  # TODO might not work like this.
        for t in range(self.num_time_steps):
            # setting min of (0, Cons_C(t) - R'_C,t)
            self.model.addConstr(self.heatingConsumption[t][0] <= 0)
            self.model.addConstr(self.heatingConsumption[t][0] <= Cons_C[t] - sum(
                self.powerModel[:, t]))  # assuming sum works as intended.

            # setting max of (0, Cons_C(t) - R'_C,t)
            self.model.addConstr(self.heatingConsumption[t][1] >= 0)
            self.model.addConstr(self.heatingConsumption[t][1] >= Cons_C[t] - sum(
                self.powerModel[:, t]))  # assuming sum works as intended.

    def set_peak_charge(self):
        """Sets the peak demand consumption over all zones and all times in the given time frame.
         For any further computation with power, just divide by self.delta_time."""
        self.model.addConstr(self.peakCharge >= 0)  # added to make sure we never go below 0 and we are only charged.
        self.model.addConstr(self.peakCharge >= self.last_peak_demand)  # Can't get lower than the last peak consumption.

        for z in range(self.num_zones):
            for t in range(self.num_time_steps):
                # TODO Check if this is how we want to model the exchange. Thanos gives ok.
                self.model.addConstr(self.peakCharge >= self.heatingConsumption[z][t] - self.powerModel[z][t])

    def set_temperature_in(self):
        """Set the temperatureIn in self.model for the linear program as it depends on both self.heat and self.cool.
        Formula: $$$T^{in}(zone, time) = T^{in}(zone, 0)*(\prod_{i=0}^{t-1}\gamma^{time}(zone)) + 
                    \sum_{i=0}^{time-1} \epsilon(zone, i)* \prod_{k=i+1}^{t-1}\gamma(zone, k)$$$
        where $$$\epsilon(zone, time) = (self.maxPower(zone, time)*self.action(zone, time) + 
                self.leakage_rate(zone, time)*T^{out}(zone, time))*self.delta_time$$$
        and where $$$\gamma(zone, time) = (1-self.leakage_rate(zone,time)) * self.delta_time$$$"""

        def get_gamma(zone, time):
            return 1 - self.leakage_rate[zone][time] * self.delta_time

        def get_epsilon(zone, time):
            # return epsilon adjusted for which self.action we are using.
            return (self.maxPower[zone][time] * self.action[zone][time] + \
                    self.leakage_rate[zone][time] * self.t_out[zone][time]) * self.delta_time

        # stores the last epsilon and gamma values so i don't have to recompute them
        last_gamma_temperature = np.array([self.t_in[z][0] * get_gamma(z, 0) for z in range(self.num_zones)])
        last_epsilon_gamma_sum = np.array([get_epsilon(z, 0) for z in range(self.num_zones)])

        for z in range(self.num_zones):
            for t in range(1, self.num_time_steps):
                self.model.addConstr(self.t_in[z][t] ==
                                     last_epsilon_gamma_sum[z] + last_gamma_temperature[z])
                # setting the terms for the next constraint. Follows naturally from recursive form from the paper
                # "AEC of Electricity-based Space Heating Systems".
                last_epsilon_gamma_sum[z] = last_epsilon_gamma_sum[z] * get_gamma(z, t) + get_epsilon(z, t)
                last_gamma_temperature[z] *= get_gamma(z, t)

    def setTemperatureSetpoint(self):
        self.temperatureSetpoint = "TODO"
