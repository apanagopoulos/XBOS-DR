from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe

import networkx as nx
import numpy as np
import pandas as pd

from ThermalModel import ThermalModel
from Occupation import Occupation
from Discomfort import Discomfort
from Safety import Safety
from EnergyConsumption import EnergyConsumption
import datetime
import pytz

#this is a Node of the graph
class Node:
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

# The EVA class contains the shortest path algorithm and its utility functions 
class EVA:
	def __init__(self, current_time, l, pred_window, interval, interval_demand, interval_usage,
				root=Node([75], 0), starting_max_demand=0, noZones=1, discomfort=None,
				thermal=None, occupation=None, safety=None, energy=None):

		# initialize class constants
		self.noZones = noZones
		self.current_time = current_time
		self.l = l
		self.g = nx.DiGraph()
		self.interval = interval
		self.root = root
		self.target = self.get_real_time(pred_window * interval)
		self.interval_demand = interval_demand
		self.interval_usage = interval_usage

		self.billing_period = 30 * 24 * 60 / interval  # 30 days

		self.disc = discomfort if (discomfort is not None) else Discomfort()
		self.th = thermal if (thermal is not None) else ThermalModel()
		self.occ = occupation if (occupation is not None) else Occupation()
		self.safety = safety if (safety is not None) else Safety(noZones=self.noZones)
		self.energy = energy if (energy is not None) else EnergyConsumption()

		self.starting_max_demand = starting_max_demand
		self.g.add_node(root, usage_cost=np.inf, best_action=None, max_demand=np.inf)

	# util function that convers from integer that converts from relevant time to real time
	def get_real_time(self, node_time):
		return self.current_time + datetime.timedelta(minutes=node_time)

	# the shortest path algorithm
	def shortest_path(self, from_node):
		"""
		Creates the graph using DFS while we determine the best path
		:param from_node: the node we are currently on
		"""

		# add the final nodes when algorithm goes past the target prediction time 
		if self.get_real_time(from_node.time) >= self.target:
			self.g.add_node(from_node, usage_cost=0, best_action=None, max_demand=self.starting_max_demand)
			return

		#create the action set (0 is for do nothing, 1 is for cooling, 2 is for heating)
		action_set = self.safety.safety_actions(from_node.temps)
		
		#iterate for each available action
		for action in action_set:
			
			# predict temperature and energy consumption of action
			new_temperature = []
			consumption = []
			for i in range(self.noZones):
				new_temperature.append(self.th.next_temperature(from_node.temps[i], action[i], zone=i))
				consumption.append(self.energy.calc_cost(action[i], from_node.time))
				
			if self.safety.safety_check(new_temperature) and len(action_set) > 1:
				continue

			# create the node that describes the predicted data
			new_node = Node(
				temps=new_temperature,
				time=from_node.time + self.interval
			)

			# calculate interval costs
			demand_balancer = (((self.target - self.get_real_time(from_node.time/self.interval)).total_seconds() / 60) / self.billing_period)
			interval_usage_cost = sum(consumption) + self.interval_usage[int(from_node.time/self.interval)]
			interval_demand = self.interval_demand[int(from_node.time/self.interval)]

			#calculate interval discomfort
			discomfort = [0] * self.noZones

			for i in range(self.noZones):
				discomfort[i] = self.disc.disc(new_temperature[i], self.occ.occ(from_node.time/self.interval), from_node.time)

			# create node if the new node is not allready in graph
			# recursively run shortest path for the new node
			if new_node not in self.g:
				self.g.add_node(new_node, usage_cost=np.inf, best_action=None, max_demand=np.inf)
				self.shortest_path(new_node)

			# calculate path costs
			this_path_demand = max(self.g.node[new_node]['max_demand'], interval_demand)
			this_path_demand_cost = demand_balancer * this_path_demand

			interval_overall_cost = ((1 - self.l) * (interval_usage_cost + this_path_demand_cost)) + (self.l * (sum(discomfort)))

			this_path_cost = self.g.node[new_node]['usage_cost'] + interval_overall_cost

			# add the edge connecting this state to the previous
			self.g.add_edge(from_node, new_node, action=action)

			# choose the shortest path
			if this_path_cost <= self.g.node[from_node]['usage_cost']:
				if this_path_cost == self.g.node[from_node]['usage_cost'] and self.g.node[from_node]['best_action'] == '0':
					continue
				self.g.add_node(from_node, best_action=new_node, usage_cost=this_path_cost, max_demand=this_path_demand)

	# util function that recostructs the best action path
	def reconstruct_path(self, graph=None):
		if graph == None:
			graph = self.g

		cur = self.root
		path = [cur]

		while graph.node[cur]['best_action'] is not None:
			cur = graph.node[cur]['best_action']
			path.append(cur)

		return path

# the Advise class initializes all the 
class Advise:
	def __init__(self):

		# initialize constants
		Lambda = 0.995
		Interval_Length = 15  # in minutes
		Hours = 4
		Intervals = Hours * 60 / Interval_Length
		Predicted_Interval_Demands = [0]*Intervals
		Predicted_Interval_Usage = [0]*Intervals
		No_Of_Zones = 1
		Maximum_Safety_Temp, Minimum_Safety_Temp = 86, 54
		Heating_Consumption = 4000  # watts
		Cooling_Consumption = 4000  # watts
		
		max_demand = 0
		self.current_time = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
		# collect data for the thermal model
		self.df = self.inside_temperature()
		
		# initialize all models
		disc = Discomfort(now=self.current_time)
		th = ThermalModel(now=self.current_time)
		occ = Occupation(now=self.current_time)
		safety = Safety(max_temperature=Maximum_Safety_Temp, min_temperature=Minimum_Safety_Temp, noZones=No_Of_Zones)
		energy = EnergyConsumption(now=self.current_time, heat=Heating_Consumption, cool=Cooling_Consumption)
		
		Zones_Starting_Temps = [self.df[-1]]
		self.root = Node(Zones_Starting_Temps, 0)

		# initialize the shortest path model
		self.advise_unit = EVA(
			current_time=self.current_time,
			l=Lambda,
			pred_window=Intervals,
			interval= Interval_Length,
			interval_demand= Predicted_Interval_Demands,
			interval_usage= Predicted_Interval_Usage,
			root=self.root,
			starting_max_demand=max_demand,
			noZones=No_Of_Zones,
			discomfort=disc,
			thermal=th,
			occupation=occ,
			safety=safety,
			energy=energy
		)	

	def inside_temperature(self):

		#c = get_client(agent = '172.17.0.1:28589', entity="thanos.ent")
		c = get_client()
		archiver = DataClient(c)

		se_temp = "b47ba370-bceb-39cf-9552-d1225d910039"
		se_state = "7e543d07-16d1-32bb-94af-95a01f4675f9"

		uuids = [se_temp, se_state]

		temp_now = self.current_time

		start = '"' + temp_now.strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
		end = '"' + (temp_now - datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'

		dfs = make_dataframe(archiver.window_uuids(uuids, end, start, '1min', timeout=120))

		for uid, df in dfs.items():
			
			if uid == uuids[0]:
				if 'mean' in df.columns:
					df = df[['mean']]
				df.columns = ['tin']
				
			dfs[uid] = df.resample('1min').mean()

		uid, df = dfs.items()[0]
		df['tin']=df['tin'].replace(to_replace=0, method='pad')

		return df['tin']

	def advise(self):
		self.advise_unit.shortest_path(self.root)
		path = self.advise_unit.reconstruct_path()
		action = self.advise_unit.g[path[0]][path[1]]['action']
		#max_demand = self.advise_unit.g.node[path[0]]['max_demand']

		return action, self.df[-1]

