import datetime
import os

import networkx as nx
import plotly.offline as py
import pytz
import yaml
import numpy as np

import sys
sys.path.insert(0, '../')
import utils
from utils import plotly_figure

from Discomfort import Discomfort
from EnergyConsumption import EnergyConsumption
from Occupancy import Occupancy
from Safety import Safety
from ThermalModel import ThermalModel


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
        return isinstance(other, self.__class__) \
               and self.temps == other.temps \
               and self.time == other.time

    def __repr__(self):
        return "{0}-{1}".format(self.time, self.temps)


class EVA:
    def __init__(self, current_time, l, pred_window, interval, discomfort,
                 thermal, occupancy, safety, energy, zones, root=Node([75], 0), noZones=1):
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
        """
        # initialize class constants
        # Daniel added TODO Seriously come up with something better to handle how assign zone to predict. Right now only doing it this way because we want to generalize to multiclass.
        self.zones = zones

        self.noZones = noZones
        self.current_time = current_time
        self.l = l
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
            self.g.add_node(from_node, usage_cost=0, best_action=None, best_successor=None, discomfort=None, consumption=None)
            return

        # create the action set (0 is for do nothing, 1 is for cooling, 2 is for heating)
        action_set = self.safety.safety_actions(from_node.temps, from_node.time / self.interval)

        # iterate for each available action
        # actions are strings.
        for action in action_set:

            # predict temperature and energy cost of action
            new_temperature = []
            consumption = []
            for i in range(self.noZones):
                # Note: we are assuming self.zones and self.temps are in right order.
                new_temperature.append(self.th.predict(t_in=from_node.temps[i],
                                                       zone=self.zones[i],
                                                       action=int(action[i]),
                                                       time=self.get_real_time(from_node.time).hour)[
                                           0])  # index because self.th.predict returns array.
                consumption.append(self.energy.calc_cost(action[i], from_node.time / self.interval))

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
                self.g.add_node(new_node, usage_cost=np.inf, best_action=None, best_successor=None, discomfort=None, consumption=None)
                self.shortest_path(new_node)

            # need to find a way to get the consumption and discomfort values between [0,1]
            interval_overall_cost = ((1 - self.l) * (sum(consumption))) + (self.l * (sum(discomfort)))

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
                 thermal_precision, occ_obs_len_addition, setpoints, sensors, safety_constraints):
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
        self.root = Node(Zones_Starting_Temps, 0)
        temp_l = dr_lamda if dr else lamda

        print("Lambda being used for zone %s is of value %s" % (zones[0], str(temp_l)))

        # initialize the shortest path model
        self.advise_unit = EVA(
            current_time=self.current_time,
            l=temp_l,
            pred_window=predictions_hours * 60 / interval,
            interval=interval,
            discomfort=disc,
            thermal=thermal_model,
            occupancy=occ,
            safety=safety,
            energy=energy,
            root=self.root,
            zones=zones
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
    import Debugger
    from DataManager import DataManager

    from xbos import get_client
    from ControllerDataManager import ControllerDataManager
    from AverageThermalModel import AverageMPCThermalModel

    from xbos.services.hod import HodClient
    from xbos.devices.thermostat import Thermostat
    import time

    ZONE = "HVAC_Zone_AC-1"
    building = sys.argv[1]
    naive_now = datetime.datetime.strptime("2018-06-22 15:20:44", "%Y-%m-%d %H:%M:%S")
    now =  pytz.timezone("UTC").localize(naive_now)

    # TODO check for comfortband height and whether correctly implemented
    # read from config file
    try:
        yaml_filename = "../Buildings/%s/%s.yml" % (sys.argv[1], sys.argv[1])
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
        with open("../Thermal Data/demo_" + cfg["Building"], "r") as f:
            thermal_data = pickle.load(f)
    except:
        controller_dataManager = ControllerDataManager(cfg, client)
        thermal_data = controller_dataManager.thermal_data(days_back=50)
        with open("../Thermal Data/demo_" + cfg["Building"], "wb") as f:
            pickle.dump(thermal_data, f)

    # Concat zone data to put all data together and filter such that all datapoints have dt != 1
    building_thermal_data = utils.concat_zone_data(thermal_data)
    filtered_building_thermal_data = building_thermal_data[building_thermal_data["dt"]!=1]


    # TODO INTERVAL SHOULD NOT BE IN config_file.yml, THERE SHOULD BE A DIFFERENT INTERVAL FOR EACH ZONE
    # TODO, NOTE, We are training on the whole building.
    zone_thermal_models = {zone: AverageMPCThermalModel(zone, filtered_building_thermal_data, interval_length=cfg["Interval_Length"],
                                                 thermal_precision=cfg["Thermal_Precision"])
                           for zone, zone_thermal_data in thermal_data.items()}
    print("Trained Thermal Model")
    # --------------------------------------

    with open("../Buildings/" + cfg["Building"] + "/ZoneConfigs/" + ZONE + ".yml", 'r') as ymlfile:
        advise_cfg = yaml.load(ymlfile)


    dataManager = DataManager(cfg, advise_cfg, client, ZONE, now=now)
    safety_constraints = dataManager.safety_constraints()
    prices = dataManager.prices()
    building_setpoints = dataManager.building_setpoints()

    temperature = 67.8
    DR = False

    adv = Advise([ZONE],  # array because we might use more than one zone. Multiclass approach.
                 now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])),
                 dataManager.preprocess_occ(),
                 [temperature],
                 zone_thermal_models[ZONE],
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

    Debugger.debug_print(now, building, ZONE, adv, safety_constraints, prices, building_setpoints, adv_end - adv_start, file=False)
    adv.g_plot(ZONE)

