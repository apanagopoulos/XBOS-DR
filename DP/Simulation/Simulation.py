import sys, glob, os, pickle, pytz, datetime
import numpy as np
from datetime import timedelta
import pandas as pd
sys.path.insert(0, '../Server')
sys.path.insert(0, '../Server/MPC')
sys.path.insert(0, '../Server/MPC/ThermalModels')
from Advise import Advise
from MPCThermalModel import *
from DataManager import DataManager
from ThermalDataManager import ThermalDataManager
import utils
from xbos import get_client
from xbos.services.hod import HodClient
from xbos.services import mdal

#calculate real discomfort
def calc_discomfort(t_in, setpoints, occ, interval):
	if t_in > setpoints[0] and t_in < setpoints[1]:
		return 0
	else:
		if abs(setpoints[0] - t_in) < abs(setpoints[1] - t_in):
			return ((setpoints[0] - t_in) ** 2.) * interval * occ
		else:
			return ((setpoints[1] - t_in) ** 2.) * interval * occ

#calculate real cost
def calc_cost(action, price, advise_cfg, period):
	if action == 'Heating' or action == '2':
		return (advise_cfg['Advise']['Heating_Consumption'] * float(period) / 60.) * price
	elif action == 'Cooling' or action == '1':
		return (advise_cfg['Advise']['Cooling_Consumption'] * float(period) / 60.) * price
	elif action == 'Do Nothing' or action == '0':
		return 0
	else:
		print("picked wrong action")
		return 0

def getDatetime(date_string):
	"""Gets datetime from string with format HH:MM.
	:param date_string: string of format HH:MM
	:returns datetime.time() object with no associated timzone. """
	return datetime.datetime.strptime(date_string, "%H:%M").time()

#takes datetime.datetime.time() objects as input, returns True if now is between start and end
def in_between(now, start, end):
	if start < end:
		return start <= now < end
	elif end < start:
		return start <= now or now < end
	else:
		return True

#this class should handle the thermal models
class Thermal:

	def __init__(self, cfg, client, now):

		self.now = now #starting time
		self.cfg = cfg #building config
		self.current_day = 0 #days after start

		#train thermal models:
		thermal_data = utils.get_data(cfg=cfg, client=client, days_back=150, force_reload=False)
		self.zones = [zone for zone, zone_thermal_data in thermal_data.items()]
		self.zone_thermal_models = {}

		for zone, zone_data in thermal_data.items():
		# Concat zone data to put all data together and filter such that all datapoints have dt != 1
			filtered_zone_data = zone_data[zone_data["dt"] == 5]
			self.zone_thermal_models[zone] = MPCThermalModel(zone=zone, thermal_data=filtered_zone_data,
											interval_length=cfg["Interval_Length"],
											thermal_precision=cfg["Thermal_Precision"])
		#building_thermal_data = utils.concat_zone_data(thermal_data)
		#filtered_building_thermal_data = building_thermal_data[building_thermal_data["dt"] != 1]

		"""self.zone_thermal_models = {
		zone: AverageMPCThermalModel(zone, filtered_building_thermal_data, interval_length=cfg["Interval_Length"],
									 thermal_precision=cfg["Thermal_Precision"])
		for zone, zone_thermal_data in thermal_data.items()}
		"""

		self.weather_predictions = self.weather_fetch() #weather predictons for the next 4 days
		self.weather = self.weather_predictions[self.current_day] #weather predictions for the next 24 hours
		for i in self.zones:
			self.zone_thermal_models[i].set_weather_predictions(self.weather)

		print("Trained Thermal Models")

	#updates the weather for each passing hour
	def update_weather(self, hour_to_change):

		self.weather[hour_to_change] = self.weather_predictions[self.current_day+1][hour_to_change]
		for i in self.zones:
			self.zone_thermal_models[i].set_weather_predictions(self.weather)

	#collects the weather predictions for the next 4 days
	def weather_fetch(self):
		import requests
		import json
		from dateutil import parser
		file_name = self.cfg["Building"] + "_weather.json"
		coordinates = self.cfg["Coordinates"]
		fetch_attempts = 3

		weather_fetch_successful = False
		while not weather_fetch_successful and fetch_attempts > 0:
			if not os.path.exists(file_name):
				temp = requests.get("https://api.weather.gov/points/" + coordinates).json()
				weather = requests.get(temp["properties"]["forecastHourly"])
				data = weather.json()
				with open(file_name, 'wb') as f:
					json.dump(data, f)

			try:
				with open(file_name, 'r') as f:
					myweather = json.load(f)
					weather_fetch_successful = True
			except:
				# Error with reading json file. Refetching data.
				print("Warning, received bad weather.json file. Refetching from archiver.")
				os.remove(file_name)
				weather_fetch_successful = False
				fetch_attempts -= 1

		if fetch_attempts == 0:
			raise Exception("ERROR, Could not get good data from weather service.")

		# got an error on parse in the next line that properties doesnt exit
		json_start = parser.parse(myweather["properties"]["periods"][0]["startTime"])
		if (json_start.hour < self.now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"])).hour) or \
				(datetime.datetime(json_start.year, json_start.month, json_start.day).replace(
					tzinfo=pytz.timezone(self.cfg["Pytz_Timezone"])) <
				 datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(
					 tz=pytz.timezone(self.cfg["Pytz_Timezone"]))):
			temp = requests.get("https://api.weather.gov/points/" + coordinates).json()
			weather = requests.get(temp["properties"]["forecastHourly"])
			data = weather.json()
			with open(file_name, 'w') as f:
				json.dump(data, f)
			myweather = json.load(open(file_name))

		weather_predictions = {}

		for i in range(4):
			weather_predictions[i] = {}
		j = 0
		for i, data in enumerate(myweather["properties"]["periods"]):
			hour = parser.parse(data["startTime"]).hour
			weather_predictions[j][hour] = int(data["temperature"])

			if i>0 and i%23==0:
				j += 1
			if j > 3:
				break

		return weather_predictions

# this is the occupancy handling class
#TODO OCCUPANCY CLASS NEEDS REVAMP AND A LOT OF CHANGES
class Occupancy:

	def __init__(self, client, hod_client, cfg, advise_cfgs, zones, now):
		self.c = client
		self.hc = hod_client
		self.cfg = cfg #building config
		self.zones = zones #list of zone names
		self.advise_cfgs = advise_cfgs #zone configs
		self.now = now #starting time
		self.occ_array = self.preprocess_occ()

	# fetches historical data if sensors exists, creates occupancy array for the next hours if it doesnt
	def preprocess_occ(self):

		occ_flag = False
		for zone in self.zones:
			if self.advise_cfgs[zone]["Advise"]["Occupancy_Sensors"] == True:
				occ_flag = True

		if occ_flag:
			occ_query = """SELECT ?sensor ?uuid ?zone FROM %s WHERE {
						  ?sensor rdf:type brick:Occupancy_Sensor .
						  ?sensor bf:isPointOf/bf:isPartOf ?zone .
						  ?sensor bf:uuid ?uuid .
						  ?zone rdf:type brick:HVAC_Zone};""" % self.cfg["Building"]
			# get all the occupancy sensors uuids
			results = self.hc.do_query(occ_query)  # run the query
			uuids = [[x['?zone'], x['?uuid']] for x in results['Rows']]  # unpack
			c = mdal.MDALClient("xbos/mdal", client=self.c)
		zone_occupancies = {}

		for zone in self.zones:

			if self.advise_cfgs[zone]["Advise"]["Occupancy_Sensors"]:

				# only choose the sensors for the zone specified in cfg
				query_list = []
				for i in uuids:
					if i[0] == zone:
						query_list.append(i[1])

				# get the sensor data

				dfs = c.do_query({'Composition': query_list,
								  'Selectors': [mdal.MAX] * len(query_list),
								  'Time': {'T0': (self.now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
										   'T1': self.now.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
										   'WindowSize': str(self.cfg["Interval_Length"]) + 'min',
										   'Aligned': True}})

				dfs = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)

				df = dfs[[query_list[0]]]
				df.columns.values[0] = 'occ'
				df.is_copy = False
				df.columns = ['occ']
				# perform OR on the data, if one sensor is activated, the whole zone is considered occupied
				for i in range(1, len(query_list)):
					df.loc[:, 'occ'] += dfs[query_list[i]]
				df.loc[:, 'occ'] = 1 * (df['occ'] > 0)

				zone_occupancies[zone] = df.tz_localize(None)
			else:
				occupancy_array = self.advise_cfgs[zone]["Advise"]["Occupancy"]

				now_time = self.now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"]))
				occupancy = []

				while now_time <= self.now + timedelta(hours=self.advise_cfgs[zone]["Advise"]["MPCPredictiveHorizon"]):
					i = now_time.weekday()

					for j in occupancy_array[i]:
						if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
									  datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
							occupancy.append(j[2])
							break

					now_time += timedelta(minutes=self.cfg["Interval_Length"])

					zone_occupancies[zone] = occupancy

		return zone_occupancies

	#updates the historical data if sensors exist, creates the new occupancy array if it doesnt
	def update_occ_array(self, new_occs):

		for zone in self.zones:
			if self.advise_cfgs[zone]["Advise"]["Occupancy_Sensors"]:
				self.occ_array[zone] = self.occ_array[zone].append(pd.DataFrame(new_occs[zone],
														 index = self.occ_array[zone].tail(1).index \
																 + timedelta(minutes=int(self.cfg["Interval_Length"])),
														 columns=['occ'])).iloc[1:]
			else:
				self.occ_array[zone] = self.preprocess_occ()[zone]

#this is the simulaton class
class Simulation:

	def __init__(self, building, hours=24, cfg=None, plot_friendly_output=True):

		self.plo = plot_friendly_output #True is per minute save, False is per interval save
		self.hours = hours #how many hours ahead to simulate
		if hours > 72:
			raise Exception("Hours must be 72 or less")
		self.building = building
		if cfg is None:
			yaml_filename = "../Server/Buildings/%s/%s.yml" % (building, building)
			with open(yaml_filename, 'r') as ymlfile:
				self.cfg = yaml.load(ymlfile)
		else:
			self.cfg = cfg

		if self.cfg["Server"]:
			self.client = get_client(agent=self.cfg["Agent_IP"], entity=self.cfg["Entity_File"])
		else:
			self.client = get_client()

		hc = HodClient("xbos/hod", self.client)
		self.now = pytz.timezone("UTC").localize(datetime.datetime.utcnow())

		self.t_models = Thermal(self.cfg, self.client, self.now)
		self.tstats = utils.get_thermostats(self.client, hc, self.cfg["Building"])
		self.zones = [zone for zone, tstat in self.tstats.items()]
		self.starting_temperatures = {dict_zone: dict_tstat.temperature for dict_zone, dict_tstat in self.tstats.items()}
		self.advise_cfg = {}
		for zone in self.zones:
			with open("../Server/Buildings/%s/ZoneConfigs/%s.yml" % (building, zone), 'r') as ymlfile:
				self.advise_cfg[zone] = yaml.load(ymlfile)

		self.Occ = Occupancy(self.client, hc, self.cfg, self.advise_cfg, self.zones, self.now)

	def run(self):

		now_time = self.now
		current_hour = now_time.hour

		temp_now = self.starting_temperatures.copy()
		next_temp = temp_now

		total_occupancy = {}
		total_disc = {}
		total_cost = {}
		total_temp = {}
		total_action = {}
		total_price = {}
		total_heating_stp = {}
		total_cooling_stp = {}

		for zone in self.zones:
			total_occupancy[zone] = []
			total_disc[zone] = []
			total_cost[zone] = []
			total_temp[zone] = []
			total_action[zone] = []
			total_price[zone] = []
			total_heating_stp[zone] = []
			total_cooling_stp[zone] = []

		while now_time < self.now + timedelta(hours=self.hours):

			temp_now = next_temp.copy()
			occupancy_dict = {}

			#this handles the weather on the thermal model
			if now_time.hour != current_hour:
				if (now_time - self.now).days > self.t_models.current_day:
					self.t_models.current_day = (now_time - self.now).days
				self.t_models.update_weather(current_hour)
				current_hour = now_time.hour

			#run the mpc for each zone
			for zone in self.zones:

				prices = self.prices(now_time, self.advise_cfg[zone])
				building_setpoints = self.building_setpoints(now_time, self.advise_cfg[zone])
				safety_constraints = self.safety_constraints(now_time, self.advise_cfg[zone])
				self.t_models.zone_thermal_models[zone].zoneTemperatures = temp_now.copy()
				adv = Advise([zone],  # array because we might use more than one zone. Multiclass approach.
							 self.now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"])),
							 self.Occ.occ_array[zone],
							 [temp_now[zone]],
							 self.t_models.zone_thermal_models[zone],
							 prices,
							 self.advise_cfg[zone]["Advise"]["General_Lambda"],
							 self.advise_cfg[zone]["Advise"]["DR_Lambda"],
							 False,
							 self.cfg["Interval_Length"],
							 self.advise_cfg[zone]["Advise"]["MPCPredictiveHorizon"],
							 self.advise_cfg[zone]["Advise"]["Heating_Consumption"],
							 self.advise_cfg[zone]["Advise"]["Cooling_Consumption"],
							 self.advise_cfg[zone]["Advise"]["Ventilation_Consumption"],
							 self.advise_cfg[zone]["Advise"]["Thermal_Precision"],
							 self.advise_cfg[zone]["Advise"]["Occupancy_Obs_Len_Addition"],
							 building_setpoints,
							 self.advise_cfg[zone]["Advise"]["Occupancy_Sensors"],
							 safety_constraints)

				#handle the occupancy updates
				if self.advise_cfg[zone]["Advise"]["Occupancy_Sensors"]:
					occupancy_dict[zone] = 1 if adv.occ_predictions.iloc[1][0]>=0.5 else 0
					prediction = adv.occ_predictions.iloc[0][0]
				else:
					occupancy_dict[zone] = None
					prediction = adv.occ_predictions[0]

				action = adv.advise() #get action
				next_temp[zone] = self.t_models.zone_thermal_models[zone].predict(t_in=temp_now[zone], action=int(action), time= now_time.hour)

				#save this zone data for this interval
				if self.plo:
					for j in range(self.cfg["Interval_Length"]):
						disc = calc_discomfort(temp_now[zone] + (next_temp[zone] - temp_now[zone]) * j / float(self.cfg["Interval_Length"]),
											   building_setpoints[0], prediction, 1)
						total_disc[zone].append(disc)
						cost = calc_cost(action, prices[0], self.advise_cfg[zone], 1)
						total_cost[zone].append(cost)
						total_occupancy[zone].append(prediction)
						total_temp[zone].append(temp_now[zone] + (next_temp[zone] - temp_now[zone]) * j / float(self.cfg["Interval_Length"]))
						total_action[zone].append(action)
						total_price[zone].append(prices[0])
						total_heating_stp[zone].append(building_setpoints[0][0])
						total_cooling_stp[zone].append(building_setpoints[0][1])
				else:
					disc = calc_discomfort(next_temp[zone], building_setpoints[0], prediction, self.cfg["Interval_Length"])
					total_disc[zone].append(disc)
					cost = calc_cost(action, prices[0], self.advise_cfg[zone], self.cfg["Interval_Length"])
					total_cost[zone].append(cost)
					total_occupancy[zone].append(prediction)
					total_temp[zone].append(next_temp[zone])
					total_action[zone].append(action)
					total_price[zone].append(prices[0])
					total_heating_stp[zone].append(building_setpoints[0][0])
					total_cooling_stp[zone].append(building_setpoints[0][1])

			#update the occupancy data
			self.Occ.update_occ_array(occupancy_dict)

			now_time += timedelta(minutes=self.cfg["Interval_Length"])

			print now_time
			print temp_now

		#print the simulation output
		print "Starting temp:"
		print self.starting_temperatures
		print "Finishing temp:"
		print next_temp
		print "Simulation is finished"
		for zone in self.zones:
			print "Total cost: " + str(np.sum(total_cost[zone]))
			print "Total disc: " + str(np.sum(total_disc[zone]))
			print "Actions where: "
			print total_action[zone]

		#save the simulation output with pickle in "outfile_buildingName.dict" format
		return_dict = {}
		return_dict["occupancy"] = total_occupancy
		return_dict["discomfort"] = total_disc
		return_dict["cost"] = total_cost
		return_dict["temperature"] = total_temp
		return_dict["action"] = total_action
		return_dict["price"] = total_price
		return_dict["heating_stp"] = total_heating_stp
		return_dict["cooling_stp"] = total_cooling_stp

		with open('outfile_'+ self.building + '.dict', 'wb') as fp:
			pickle.dump(return_dict, fp)

	#handles the pricing array for the mpc
	def prices(self, now, advise_cfg):

		pred_horizon = advise_cfg["Advise"]["MPCPredictiveHorizon"]
		price_array = self.cfg["Pricing"][self.cfg["Pricing"]["Energy_Rates"]]

		if self.cfg["Pricing"]["Energy_Rates"] == "Server":
			# not implemented yet, needs fixing from the archiver
			# (always says 0, problem unless energy its free and noone informed me)
			raise ValueError('SERVER MODE IS NOT YET IMPLEMENTED FOR ENERGY PRICING')
		else:
			now_time = now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"]))
			pricing = []

			DR_start_time = [int(self.cfg["Pricing"]["DR_Start"].split(":")[0]),
							 int(self.cfg["Pricing"]["DR_Start"].split(":")[1])]
			DR_finish_time = [int(self.cfg["Pricing"]["DR_Finish"].split(":")[0]),
							  int(self.cfg["Pricing"]["DR_Finish"].split(":")[1])]

			while now_time <= now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"])) + timedelta(hours=pred_horizon):
				i = 1 if now_time.weekday() >= 5 or self.cfg["Pricing"]["Holiday"] else 0
				if in_between(now_time.time(), datetime.time(DR_start_time[0], DR_start_time[1]),
							  datetime.time(DR_finish_time[0], DR_finish_time[1])) and \
						(self.cfg["Pricing"][
							"DR"]):  # TODO REMOVE ALLWAYS HAVING DR ON FRIDAY WHEN DR SUBSCRIBE IS IMPLEMENTED
					pricing.append(self.cfg["Pricing"]["DR_Price"])
					now_time += timedelta(minutes=self.cfg["Interval_Length"])
					continue

				for j in price_array[i]:
					if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
									datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
						pricing.append(j[2])
						break
				now_time += timedelta(minutes=self.cfg["Interval_Length"])



		return pricing

	#handles the safety constraint array for the mpc
	def safety_constraints(self, now, advise_cfg):

		pred_horizon = advise_cfg["Advise"]["MPCPredictiveHorizon"]
		setpoints_array = advise_cfg["Advise"]["SafetySetpoints"]

		def in_between(now, start, end):
			if start < end:
				return start <= now < end
			elif end < start:
				return start <= now or now < end
			else:
				return True

		now_time = now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"]))
		setpoints = []

		while now_time <= now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"])) + timedelta(hours=pred_horizon):
			i = now_time.weekday()

			for j in setpoints_array[i]:
				if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
							  datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
					setpoints.append([j[2], j[3]])
					break

			now_time += timedelta(minutes=self.cfg["Interval_Length"])

		return setpoints

	#handles the building setpoints array for the mpc
	def building_setpoints(self, now, advise_cfg):

		pred_horizon = advise_cfg["Advise"]["MPCPredictiveHorizon"]
		setpoints_array = advise_cfg["Advise"]["Comfortband"]
		safety_temperatures = advise_cfg["Advise"]["SafetySetpoints"]

		now_time = now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"]))
		setpoints = []

		while now_time <= now.astimezone(tz=pytz.timezone(self.cfg["Pytz_Timezone"])) + timedelta(hours=pred_horizon):
			weekday = now_time.weekday()

			for j in setpoints_array[weekday]:
				if in_between(now_time.time(), getDatetime(j[0]), getDatetime(j[1])) and \
						(j[2] != "None" or j[3] != "None"):  # TODO come up with better None value detection.
					setpoints.append([j[2], j[3]])
					break

				# if we have none values, replace the values with safetytemperatures.
				elif in_between(now_time.time(), getDatetime(j[0]), getDatetime(j[1])) and \
						(j[2] == "None" or j[3] == "None"):
					for safety_temperature_time in safety_temperatures[weekday]:
						if in_between(now_time.time(), getDatetime(safety_temperature_time[0]),
									  getDatetime(safety_temperature_time[1])):
							setpoints.append([safety_temperature_time[2], safety_temperature_time[3]])
							break

			now_time += timedelta(minutes=self.cfg["Interval_Length"])

		return setpoints

if __name__ == '__main__':
	#maximum hours should be 72
	#cfg is the building cfg
	#plot friendly output, if true, saves all the needed output in a per minute fashion
	#if false it saves in a per interval fashion
	sim = Simulation("avenal-veterans-hall", hours=1, cfg=None, plot_friendly_output=False)
	sim.run()