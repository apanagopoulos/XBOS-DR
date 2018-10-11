import datetime
import sys

import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.offline as py
import pytz

sys.path.insert(0, '../')
import utils
from utils import plotly_figure

from ConsumptionStorage import ConsumptionStorage
from Discomfort import Discomfort
from EnergyConsumption import EnergyConsumption
from Occupancy import Occupancy
from Safety import Safety
from DP.Server.MPC.ThermalModels.ThermalModel import ThermalModel


# TODO Demand charges right now assume constant demand charge throughout interval. should be easy to extend
# TODO but need to keep in mind that we need to then store the cost of demand charge and not the consumption
# TODO in the graph, since we actually only want to minimize cost and not consumption.


class Node:
    """
    # this is a Node of the graph for the shortest path
    """

    def __init__(self, temps, time):
        self.temps = temps
        self.time = time

    def __hash__(self):
        return hash((' '.join(str(e) for e in self.temps), self.time))

    def __eq__(self, other):
        # TODO should we make two nodes equal if not predictd by same model? Yes, so we can make use of DP. Still, ask thanos. ADD TO EDGES.
        return isinstance(other, self.__class__) \
               and self.temps == other.temps \
               and self.time == other.time

    def __repr__(self):
        return "{0}-{1}".format(self.time, self.temps)


class EVA:
    def __init__(self, start, balance_lambda, end, interval, discomfort,
                 thermal, occupancy, safety, energy, zones, consumption_storage=None,
                 root=Node([75], 0), noZones=1, debug=False):
        """
        Constructor of the Evaluation Class
        The EVA class contains the shortest path algorithm and its utility functions
        Parameters
        ----------
        start : datetime.datetime. In local timezone of zone.
        l : float (0 - 1)
        end : datetime.datetime. In local timezone of zone.
                                This time is where the leaf nodes will be which will all have None values.
        interval : int
        discomfort : Discomfort
        thermal : ThermalModel
        occupancy : Occupancy
        safety : Safety
        energy : EnergyConsumption
        root : Node
        noZones : int
        consumption_storage: If None we don't care for demand charges, otherwise provide the consumptionStorage class
                            with filled in building and tentative zone consumption.
        debug: Mainly tells us whether to store the thermal_model_type of the predictions in each node.
        """
        # initialize class constants
        # Daniel added TODO Seriously come up with something better to handle how assign zone to predict. Right now only doing it this way because we want to generalize to multiclass.
        self.zones = zones

        self.noZones = noZones
        self.start = start
        self.end = end
        self.balance_lambda = balance_lambda
        self.g = nx.DiGraph()  # [TODO:Changed to MultiDiGraph... FIX print]
        self.interval = interval
        self.root = root

        self.billing_period = 30 * 24 * 60 / interval  # 30 days

        self.disc = discomfort
        self.th = thermal
        self.occ = occupancy
        self.safety = safety
        self.energy = energy

        self.g.add_node(root, usage_cost=np.inf, best_action=None, best_successor=None)

        self.consumption_storage = consumption_storage

        self.debug = debug

    def get_real_time(self, node_time):
        """
        util function that converts from relevant time to real time
        Parameters
        ----------
        node_time : int

        Returns
        -------
        int
        """
        return self.start + datetime.timedelta(minutes=node_time)

    # the shortest path algorithm
    def shortest_path(self, from_node):
        """
        Creates the graph using DFS and calculates the shortest path

        Parameters
        ----------
        from_node : node being examined right now
        """

        # add the final nodes when algorithm goes past the target prediction time
        if self.get_real_time(from_node.time) >= self.end:
            self.g.add_node(from_node, usage_cost=0, best_action=None, best_successor=None, discomfort=[None],
                            consumption=[None], max_consumption=0)
            return

        # create the action set (0 is for do nothing, 1 is for cooling, 2 is for heating)
        action_set = self.safety.safety_actions(from_node.temps, from_node.time / self.interval)

        # iterate for each available action
        # actions are strings.
        for action in action_set:

            # predict temperature and energy cost of action
            new_temperature = []
            if self.debug:
                model_type = []
            consumption_cost = []
            zone_consumption = []
            for i in range(self.noZones):
                # Note: we are assuming self.zones and self.temps are in right order.
                # To get model types.
                if self.debug:
                    temperatures, model_types = self.th.predict(t_in=from_node.temps[i],
                                                                action=int(action[i]),
                                                                time=self.get_real_time(from_node.time).hour,
                                                                debug=True)
                    model_type.append(model_types[0])  # index because self.th.predict returns array.
                else:
                    temperatures = self.th.predict(t_in=from_node.temps[i],
                                                   action=int(action[i]),
                                                   time=self.get_real_time(from_node.time).hour, debug=False)

                new_temperature.append(temperatures[0])  # index because self.th.predict returns array.
                consumption_cost.append(self.energy.calc_cost(action[i], from_node.time / self.interval))
                zone_consumption.append(self.energy.get_consumption(action, interval=self.interval))

            # create the node that describes the predicted data
            new_node = Node(
                temps=new_temperature,
                time=from_node.time + self.interval
            )

            if self.safety.safety_check(new_temperature, new_node.time / self.interval) and len(action_set) > 1:
                continue

            # calculate interval discomfort
            discomfort = [0] * self.noZones

            for i in range(self.noZones):
                discomfort[i] = self.disc.disc((from_node.temps[i] + new_temperature[i]) / 2.,
                                               self.occ.occ(from_node.time / self.interval),
                                               from_node.time,
                                               self.interval)

            # create node if the new node is not already in graph
            # recursively run shortest path for the new node
            if new_node not in self.g:
                self.g.add_node(new_node, usage_cost=np.inf, best_action=None, best_successor=None, discomfort=[None],
                                consumption_cost=[None], max_consumption=None)
                self.shortest_path(new_node)

            if self.consumption_storage is not None:
                # Get current interval consumption
                # Get the consumption for the rest of the building
                rest_building_consumption = self.consumption_storage.get_rest_of_building_consumption(self.zones,
                                                                                                      self.get_real_time(
                                                                                                          from_node.time))
                current_interval_consumption = rest_building_consumption + sum(zone_consumption)

                # get max interval consumption for the path taken by the new_node
                future_max_interval_consumption = self.g.node[new_node]["max_consumption"]

                # find which consumption has been highest
                max_interval_consumption = max(future_max_interval_consumption, current_interval_consumption)
            else:
                max_interval_consumption = None

            # need to find a way to get the consumption_cost and discomfort values between [0,1]
            # TODO Fix the lambda for demand charges and multiply by cost per interval.
            interval_overall_cost = ((1 - self.balance_lambda) * (sum(consumption_cost))) + (
                self.balance_lambda * (sum(discomfort)))

            # If we care for demand charges
            if self.consumption_storage is not None:
                interval_overall_cost =+ (1 - self.balance_lambda) * max_interval_consumption

            this_path_cost = self.g.node[new_node]['usage_cost'] + interval_overall_cost

            # add the edge connecting this state to the previous
            if self.debug:
                self.g.add_edge(from_node, new_node, action=action, model_type=model_type)
            else:
                self.g.add_edge(from_node, new_node, action=action)

            # choose the shortest path
            if this_path_cost <= self.g.node[from_node]['usage_cost']:
                '''
                if this_path_cost == self.g.node[from_node]['usage_cost'] and self.g.node[from_node]['best_action'] == '0': [TODO: Is there any value in prunning here?]
                    continue
                '''
                self.g.add_node(from_node, best_action=action, best_successor=new_node, usage_cost=this_path_cost,
                                discomfort=discomfort, consumption_cost=consumption_cost,
                                max_consumption=max_interval_consumption)

    def reconstruct_path(self, graph=None):
        """
        Util function that reconstructs the best action path
        Parameters
        ----------
        graph : networkx graph

        Returns
        -------
        List
        """
        if graph is None:
            graph = self.g

        cur = self.root
        path = [cur]

        while graph.node[cur]['best_successor'] is not None:
            cur = graph.node[cur]['best_successor']
            path.append(cur)

        return path


class Advise:
    # the Advise class initializes all the Models and runs the shortest path algorithm
    def __init__(self, zones, start, occupancy_data, zone_temperature, thermal_model,
                 prices, lamda, dr_lamda, dr, interval, predictions_hours, heating_cons, cooling_cons,
                 vent_cons,
                 thermal_precision, occ_obs_len_addition, setpoints, sensors, safety_constraints, consumption_storage=None,
                 debug=False):
        """
        
        :param zones: 
        :param start: datetime timezone aware. In the local timezone of the zone.
        :param occupancy_data: 
        :param zone_temperature: 
        :param thermal_model: 
        :param prices: 
        :param lamda: 
        :param dr_lamda: 
        :param dr: 
        :param interval: 
        :param predictions_hours: 
        :param heating_cons: 
        :param cooling_cons: 
        :param vent_cons: 
        :param thermal_precision: 
        :param occ_obs_len_addition: 
        :param setpoints: 
        :param sensors: 
        :param safety_constraints: 
        :param consumption_storage:
        :param debug: 
        """
        # TODO do something with dr_lambda and vent const (they are added since they are in the config file.)
        # TODO Also, thermal_precision
        # set the time variables.
        self.start = start
        self.end = start + datetime.timedelta(hours=predictions_hours)
        self.interval = interval

        # initialize all models
        disc = Discomfort(setpoints, now=self.start)

        occ = Occupancy(occupancy_data, interval, predictions_hours, occ_obs_len_addition, sensors)
        self.occ_predictions = occ.predictions
        safety = Safety(safety_constraints, noZones=1)
        self.energy = EnergyConsumption(prices, interval, now=self.start,
                                        heat=heating_cons, cool=cooling_cons)

        Zones_Starting_Temps = zone_temperature

        # initialize root
        self.root = Node(temps=Zones_Starting_Temps, time=0)

        temp_l = dr_lamda if dr else lamda

        print("Lambda being used for zone %s is of value %s" % (zones[0], str(temp_l)))

        # initialize the shortest path model
        self.advise_unit = EVA(
            start=self.start,
            balance_lambda=temp_l,
            end=self.end,
            interval=interval,
            discomfort=disc,
            thermal=thermal_model,
            occupancy=occ,
            safety=safety,
            energy=self.energy,
            root=self.root,
            zones=zones,
            consumption_storage=consumption_storage,
            debug=debug
        )

    def advise(self):
        """
        function that runs the shortest path algorithm and returns the action produced by the mpc
        Returns
        -------
        String
        """
        self.advise_unit.shortest_path(self.root)
        self.path = self.advise_unit.reconstruct_path()
        self.graph = self.advise_unit.g
        action = self.advise_unit.g.node[self.root][
            "best_action"]  # self.advise_unit.g[path[0]][path[1]]['action'] [TODO Is this Fix correct?]
        return action

    def get_consumption(self):
        """
        Gets the consumption of the shortest path.
        :return: pd.df with timeseries index.
        """
        consumption_list = [self.energy.get_consumption(
            self.graph.node[self.path[i]]['best_action']) for i in range(len(self.path))]
        date_range = pd.date_range(start=self.start, end=self.end, freq=str(self.interval) + "T")
        return pd.Series(data=consumption_list, index=date_range)

    def g_plot(self, zone):

        try:
            os.remove('mpc_graph_' + zone + '.html')
        except OSError:
            pass

        fig = plotly_figure(self.advise_unit.g, path=self.path)
        py.plot(fig, filename='mpc_graph_' + zone + '.html', auto_open=False)


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '..')
    sys.path.insert(0, '../../Utils')
    sys.path.insert(0, './ThermalModels')
    import Debugger
    from DataManager import DataManager

    from MPC.ThermalModels.MPCThermalModel import MPCThermalModel

    import time

    choose_buildings = False
    if choose_buildings:
        building, zone = utils.choose_building_and_zone()
    else:
        building, zone = "avenal-veterans-hall", "HVAC_Zone_AC-4"

    # naive_now = datetime.datetime.strptime("2018-06-22 15:20:44", "%Y-%m-%d %H:%M:%S")
    # now =  pytz.timezone("UTC").localize(naive_now)
    now = utils.get_utc_now()

    # TODO check for comfortband height and whether correctly implemented
    # read from config file
    cfg = utils.get_config(building)

    client = utils.choose_client()  # not for server use here.

    # --- Thermal Model Init ------------
    # initialize and fit thermal model

    all_data = utils.get_data(building, force_reload=False, evaluate_preprocess=False, days_back=150)

    # TODO INTERVAL SHOULD NOT BE IN config_file.yml, THERE SHOULD BE A DIFFERENT INTERVAL FOR EACH ZONE
    # TODO, NOTE, We are training on the whole building.
    zone_thermal_models = {
        thermal_zone: MPCThermalModel(zone, zone_data[zone_data['dt'] == 5], interval_length=cfg["Interval_Length"],
                                      thermal_precision=0.05)
        for thermal_zone, zone_data in all_data.items()}
    print("Trained Thermal Model")
    # --------------------------------------

    advise_cfg = utils.get_zone_config(building, zone)

    dataManager = DataManager(cfg, advise_cfg, client, zone, now=now)
    safety_constraints = dataManager.safety_constraints()
    prices = dataManager.prices()
    building_setpoints = dataManager.building_setpoints()

    pred_horizon = 4
    start = now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"]))
    end = start + datetime.timedelta(hours=pred_horizon)

    # set consumption storage for demand charges
    consumption_timeseries = pd.date_range(start, end, freq=str(cfg["Interval_Length"]) + "T")
    consumption_data = np.ones(consumption_timeseries.shape) * 20
    building_consumption = pd.Series(data=consumption_data, index=consumption_timeseries)
    consumption_storage = ConsumptionStorage(building, num_threads=1, building_consumption=building_consumption)

    temperature = 65
    DR = False

    # set outside temperatures and zone temperatures for each zone.
    for thermal_zone, zone_thermal_model in zone_thermal_models.items():
        # hack. should be dictionary but just giving temperatures for all hours of the day.
        zone_thermal_model.set_outside_temperature([70] * 24)
        zone_thermal_model.set_temperatures({temp_thermal_zone: 70 for temp_thermal_zone in zone_thermal_models.keys()})

    adv = Advise([zone],  # array because we might use more than one zone. Multiclass approach.
                 now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])),
                 dataManager.preprocess_occ_cfg(),
                 [temperature],
                 zone_thermal_models[zone],
                 prices,
                 advise_cfg["Advise"]["General_Lambda"],
                 advise_cfg["Advise"]["DR_Lambda"],
                 DR,
                 cfg["Interval_Length"],
                 pred_horizon, # The predictive horizon
                 advise_cfg["Advise"]["Heating_Consumption"],
                 advise_cfg["Advise"]["Cooling_Consumption"],
                 advise_cfg["Advise"]["Ventilation_Consumption"],
                 advise_cfg["Advise"]["Thermal_Precision"],
                 advise_cfg["Advise"]["Occupancy_Obs_Len_Addition"],
                 building_setpoints,
                 advise_cfg["Advise"]["Occupancy_Sensors"],
                 safety_constraints,
                 consumption_storage=consumption_storage,
                 debug=True)

    adv_start = time.time()
    adv.advise()
    print(adv.get_consumption())
    adv_end = time.time()
    print("TIME: %f" % (adv_end - adv_start))
    path_length = len(adv.path)
    print([adv.advise_unit.g.node[adv.path[i]]['max_consumption'] for i in range(path_length)])
    Debugger.debug_print(now, building, zone, adv, safety_constraints, prices, building_setpoints, adv_end - adv_start, file=False)
    # adv.g_plot(zone)
