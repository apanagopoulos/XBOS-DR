import datetime
import os
import sys

import networkx as nx
import numpy as np
import plotly.offline as py
import pytz
import yaml

sys.path.insert(0, '../')
import utils
from utils import plotly_figure

from Discomfort import Discomfort
from EnergyConsumption import EnergyConsumption
from Occupancy import Occupancy
from Safety import Safety
from DP.Server.MPC.ThermalModels.ThermalModel import ThermalModel


class Node:
    """
    # this is a Node of the graph for the shortest path
    """

    def __init__(self, temps, time, model_type=None):
        self.temps = temps
        self.time = time
        # if we choose to debug in advise
        self.model_type = model_type

    def __hash__(self):
        return hash((' '.join(str(e) for e in self.temps), self.time))

    def __eq__(self, other):
        # TODO should we make two nodes equal if not predictd by same model? Yes, so we can make use of DP. Still, ask thanos.
        return isinstance(other, self.__class__) \
               and self.temps == other.temps \
               and self.time == other.time

    def __repr__(self):
        # if self.model_type is not None:
        #     return "{0}-{1}-{2}".format(self.time, self.temps, self.model_type)
        # else:
        return "{0}-{1}".format(self.time, self.temps)


class EVA:
    def __init__(self, current_time, balance_lambda, pred_window, interval, discomfort,
                 thermal, occupancy, safety, energy, zones, root=Node([75], 0), noZones=1, debug=False):
        """
        Constructor of the Evaluation Class
        The EVA class contains the shortest path algorithm and its utility functions
        Parameters
        ----------
        current_time : datetime.datetime
        l : float (0 - 1)
        pred_window : int
        interval : int
        discomfort : Discomfort
        thermal : ThermalModel
        occupancy : Occupancy
        safety : Safety
        energy : EnergyConsumption
        root : Node
        noZones : int
        debug: Mainly tells us whether to store the thermal_model_type of the predictions in each node.
        """
        # initialize class constants
        # Daniel added TODO Seriously come up with something better to handle how assign zone to predict. Right now only doing it this way because we want to generalize to multiclass.
        self.zones = zones

        self.noZones = noZones
        self.current_time = current_time
        self.balance_lambda = balance_lambda
        self.g = nx.DiGraph()  # [TODO:Changed to MultiDiGraph... FIX print]
        self.interval = interval
        self.root = root
        self.target = self.get_real_time(pred_window * interval)

        self.billing_period = 30 * 24 * 60 / interval  # 30 days

        self.disc = discomfort
        self.th = thermal
        self.occ = occupancy
        self.safety = safety
        self.energy = energy

        self.g.add_node(root, usage_cost=np.inf, best_action=None, best_successor=None)

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
        return self.current_time + datetime.timedelta(minutes=node_time)

    # the shortest path algorithm
    def shortest_path(self, from_node):
        """
        Creates the graph using DFS and calculates the shortest path

        Parameters
        ----------
        from_node : node being examined right now
        """

        # add the final nodes when algorithm goes past the target prediction time
        if self.get_real_time(from_node.time) >= self.target:
            self.g.add_node(from_node, usage_cost=0, best_action=None, best_successor=None, discomfort=[None], consumption=[None])
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
            consumption = []
            for i in range(self.noZones):
                # Note: we are assuming self.zones and self.temps are in right order.
                # To get model types.
                if self.debug:
                    temperatures, model_types = self.th.predict(t_in=from_node.temps[i],
                                                       action=int(action[i]),
                                                       time=self.get_real_time(from_node.time).hour, debug=True)
                    model_type.append(model_types[0])  # index because self.th.predict returns array.
                else:
                    temperatures = self.th.predict(t_in=from_node.temps[i],
                                                       action=int(action[i]),
                                                       time=self.get_real_time(from_node.time).hour, debug=False)

                new_temperature.append(temperatures[0])  # index because self.th.predict returns array.
                consumption.append(self.energy.calc_cost(action[i], from_node.time / self.interval))

            # create the node that describes the predicted data
            if self.debug:
                new_node = Node(
                    temps=new_temperature,
                    time=from_node.time + self.interval,
                    model_type=model_type
                )
            else:
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
                self.g.add_node(new_node, usage_cost=np.inf, best_action=None, best_successor=None, discomfort=[None], consumption=[None])
                self.shortest_path(new_node)

            # need to find a way to get the consumption and discomfort values between [0,1]
            interval_overall_cost = ((1 - self.balance_lambda) * (sum(consumption))) + (self.balance_lambda * (sum(discomfort)))

            this_path_cost = self.g.node[new_node]['usage_cost'] + interval_overall_cost

            # add the edge connecting this state to the previous
            self.g.add_edge(from_node, new_node, action=action)

            # choose the shortest path
            if this_path_cost <= self.g.node[from_node]['usage_cost']:
                '''
                if this_path_cost == self.g.node[from_node]['usage_cost'] and self.g.node[from_node]['best_action'] == '0': [TODO: Is there any value in prunning here?]
                    continue
                '''
                self.g.add_node(from_node, best_action=action, best_successor=new_node, usage_cost=this_path_cost,
                                discomfort=discomfort, consumption=consumption)

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
    def __init__(self, zones, current_time, occupancy_data, zone_temperature, thermal_model,
                 prices, lamda, dr_lamda, dr, interval, predictions_hours, heating_cons, cooling_cons,
                 vent_cons,
                 thermal_precision, occ_obs_len_addition, setpoints, sensors, safety_constraints, debug=False):
        # TODO do something with dr_lambda and vent const (they are added since they are in the config file.)
        # TODO Also, thermal_precision
        self.current_time = current_time

        # initialize all models
        disc = Discomfort(setpoints, now=self.current_time)

        occ = Occupancy(occupancy_data, interval, predictions_hours, occ_obs_len_addition, sensors)
        self.occ_predictions = occ.predictions
        safety = Safety(safety_constraints, noZones=1)
        energy = EnergyConsumption(prices, interval, now=self.current_time,
                                   heat=heating_cons, cool=cooling_cons)

        Zones_Starting_Temps = zone_temperature
        if debug:
            self.root = Node(temps=Zones_Starting_Temps, time=0, model_type=["Initial Temperature"])
        else:
            self.root = Node(temps=Zones_Starting_Temps, time=0)

        temp_l = dr_lamda if dr else lamda

        print("Lambda being used for zone %s is of value %s" % (zones[0], str(temp_l)))

        # initialize the shortest path model
        self.advise_unit = EVA(
            current_time=self.current_time,
            balance_lambda=temp_l,
            pred_window=predictions_hours * 60 / interval,
            interval=interval,
            discomfort=disc,
            thermal=thermal_model,
            occupancy=occ,
            safety=safety,
            energy=energy,
            root=self.root,
            zones=zones,
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

    from xbos import get_client
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

    client = utils.choose_client() # not for server use here.

    # --- Thermal Model Init ------------
    # initialize and fit thermal model

    all_data = utils.get_data(building, force_reload=False, evaluate_preprocess=False, days_back=150)

    # TODO INTERVAL SHOULD NOT BE IN config_file.yml, THERE SHOULD BE A DIFFERENT INTERVAL FOR EACH ZONE
    # TODO, NOTE, We are training on the whole building.
    zone_thermal_models = {thermal_zone: MPCThermalModel(zone, zone_data[zone_data['dt'] == 5], interval_length=cfg["Interval_Length"],
                                                 thermal_precision=cfg["Thermal_Precision"])
                           for thermal_zone, zone_data in all_data.items()}
    print("Trained Thermal Model")
    # --------------------------------------

    advise_cfg = utils.get_zone_config(building, zone)

    dataManager = DataManager(cfg, advise_cfg, client, zone, now=now)
    safety_constraints = dataManager.safety_constraints()
    prices = dataManager.prices()
    building_setpoints = dataManager.building_setpoints()

    temperature = 67.8
    DR = False

    # set outside temperatures and zone temperatures for each zone.
    for thermal_zone, zone_thermal_model in zone_thermal_models.items():
        zone_thermal_model.set_weather_predictions([70] * 24)
        zone_thermal_model.set_temperatures_and_fit({temp_thermal_zone: 70 for temp_thermal_zone in zone_thermal_models.keys()})

    adv = Advise([zone],  # array because we might use more than one zone. Multiclass approach.
                 now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])),
                 dataManager.preprocess_occ(),
                 [temperature],
                 zone_thermal_models[zone],
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

    adv_start = time.time()
    adv.advise()
    adv_end = time.time()
    print([n.model_type[0] for n in adv.path])
    Debugger.debug_print(now, building, zone, adv, safety_constraints, prices, building_setpoints, adv_end - adv_start, file=False)
    # adv.g_plot(ZONE)

