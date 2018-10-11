import datetime, pytz
from xbos import get_client
import yaml
import Utils
import pickle

from ThermalModel import MPCThermalModel
therm_data_file = open("zone_thermal_ciee")
therm_data = pickle.load(therm_data_file)

train = False
if train:
	mpcThermalModel = MPCThermalModel(therm_data, 1)
	mpcThermalModelFILE = open("mpcThermalModel", 'wb')
	pickle.dump(mpcThermalModel, mpcThermalModelFILE)
	mpcThermalModelFILE.close()


mpcThermalModelFILE = open("mpcThermalModel")
mpcThermalModel = pickle.load(mpcThermalModelFILE)
mpcThermalModelFILE.close()

def baseline(data, policy_array, prices_array, zones, kwh, string_extension):

	dictionary = {}
	for zone in zones:
		dictionary[zone] = {}
		dictionary[zone]["OPs"] =[]
		dictionary[zone]["Tins"] = []
		dictionary[zone]["Policy"] = []
		dictionary[zone]["TinsUP"] = []
		dictionary[zone]["TinsDOWN"] = []
		dictionary[zone]["Costs"] = []
		dictionary[zone]["Prices"] = []
		dictionary[zone]["Discomforts"] = []

	Tnext = {}
	for zone in zones:
		Tnext[zone] = data[zone]["Tin"][0]

	Simulation_Flag = False
	for i, policy in enumerate(policy_array):
		STPH = policy[1]  # same policy for all zones (TODO: FIX THAT)
		STPL = policy[0]  # same policy for all zones (TODO: FIX THAT)
		Tin = Tnext.copy()


		zone_temps_for_setter = {}
		for zone in zones:
			zone_temps_for_setter["HVAC_Zone_"+zone.lower().capitalize()] = Tin[zone]
		mpcThermalModel.set_zone_temperatures(zone_temps_for_setter)

		for zone in zones:
			if Utils.action(STPH, STPL, Tin[zone]) != 0. or data[zone]["State"][i] != 0.:
				Simulation_Flag = True

		for zone in zones:

			tout = data[zone]["Tout"][i]
			occ = data[zone]["Occ"][i]

			action = Utils.action(STPH, STPL, Tin[zone])
			price = prices_array[i]
			cost = Utils.cost(action, price, *kwh[zone])
			discomfort = Utils.discomfort(STPH, STPL, Tin[zone], occ)

			if Simulation_Flag:
				Tnext[zone] = mpcThermalModel.predict(Tin[zone],  "HVAC_Zone_"+zone.lower().capitalize(), action, tout)[0]
			else:
				try:
					Tnext[zone] = data[zone]["Tin"][i+1]
				except:
					Tnext[zone] = data[zone]["Tin"][i]

			dictionary[zone]["OPs"].append(occ)
			dictionary[zone]["Tins"].append(Tin[zone])
			dictionary[zone]["Policy"].append(action)
			dictionary[zone]["TinsUP"].append(STPH)
			dictionary[zone]["TinsDOWN"].append(STPL)
			dictionary[zone]["Costs"].append(cost)
			dictionary[zone]["Prices"].append(price)
			dictionary[zone]["Discomforts"].append(discomfort)

	for zone in zones:
		with open(zone + '_'+string_extension+'.pckl', 'wb') as fp:
			pickle.dump(dictionary[zone], fp)

def reality(data, policy_array, prices_array, zones, kwh, string_extension):


	dictionary = {}
	for zone in zones:
		dictionary[zone] = {}
		dictionary[zone]["OPs"] = []
		dictionary[zone]["Tins"] = []
		dictionary[zone]["Policy"] = []
		dictionary[zone]["TinsUP"] = []
		dictionary[zone]["TinsDOWN"] = []
		dictionary[zone]["Costs"] = []
		dictionary[zone]["Prices"] = []
		dictionary[zone]["Discomforts"] = []
		dictionary[zone]["Real_Setpoints_High"] = []
		dictionary[zone]["Real_Setpoints_Low"] = []


	for i, policy in enumerate(policy_array):
		STPH = policy[1]  # same policy for all zones (TODO: FIX THAT)
		STPL = policy[0]  # same policy for all zones (TODO: FIX THAT)
		Tin = {}
		for zone in zones:
			Tin[zone] = data[zone]["Tin"][i]

		for zone in zones:
			occ = data[zone]["Occ"][i]
			action = data[zone]["State"][i]
			price = prices_array[i]
			cost = Utils.cost(action, price, *kwh[zone])
			discomfort = Utils.discomfort(STPH, STPL, Tin[zone], occ)


			dictionary[zone]["OPs"].append(occ)
			dictionary[zone]["Tins"].append(Tin[zone])
			dictionary[zone]["Policy"].append(action)
			dictionary[zone]["TinsUP"].append(STPH)
			dictionary[zone]["TinsDOWN"].append(STPL)
			dictionary[zone]["Costs"].append(cost)
			dictionary[zone]["Prices"].append(price)
			dictionary[zone]["Discomforts"].append(discomfort)
			dictionary[zone]["Real_Setpoints_High"].append(data[zone]["STPH"][i])
			dictionary[zone]["Real_Setpoints_Low"].append(data[zone]["STPL"][i])

	for zone in zones:
		with open(zone + '_' + string_extension + '.pckl', 'wb') as fp:
			pickle.dump(dictionary[zone], fp)


def relive(data, policy_array, prices_array, zones, kwh, string_extension):
	data = data.copy()
	dictionary = {}
	for zone in zones:
		dictionary[zone] = {}
		dictionary[zone]["OPs"] = []
		dictionary[zone]["Tins"] = []
		dictionary[zone]["Policy"] = []
		dictionary[zone]["TinsUP"] = []
		dictionary[zone]["TinsDOWN"] = []
		dictionary[zone]["Costs"] = []
		dictionary[zone]["Prices"] = []
		dictionary[zone]["Discomforts"] = []
		dictionary[zone]["Real_Setpoints_High"] = []
		dictionary[zone]["Real_Setpoints_Low"] = []

	Tnext = {}
	for zone in zones:
		Tnext[zone] = data[zone]["Tin"][0]

	Simulation_Flag = False
	for i, policy in enumerate(policy_array):
		STPH = policy[1]  # same policy for all zones (TODO: FIX THAT)
		STPL = policy[0]  # same policy for all zones (TODO: FIX THAT)
		Tin = Tnext.copy()

		zone_temps_for_setter = {}
		for zone in zones:
			zone_temps_for_setter["HVAC_Zone_" + zone.lower().capitalize()] = Tin[zone]
		mpcThermalModel.set_zone_temperatures(zone_temps_for_setter)

		for zone in zones:
			if Utils.action(STPH, STPL, Tin[zone]) != 0. or data[zone]["State"][i] != 0.:
				Simulation_Flag = True

		for zone in zones:
			tout = data[zone]["Tout"][i]
			occ = data[zone]["Occ"][i]

			action = data[zone]["State"][i]
			price = prices_array[i]
			cost = Utils.cost(action, price, *kwh[zone])
			discomfort = Utils.discomfort(STPH, STPL, Tin[zone], occ)

			if Simulation_Flag:
				Tnext[zone] = mpcThermalModel.predict(Tin[zone],  "HVAC_Zone_"+zone.lower().capitalize(), action, tout)[0]
			else:
				try:
					Tnext[zone] = data[zone]["Tin"][i+1]
				except:
					Tnext[zone] = data[zone]["Tin"][i]

			dictionary[zone]["OPs"].append(occ)
			dictionary[zone]["Tins"].append(Tin[zone])
			dictionary[zone]["Policy"].append(action)
			dictionary[zone]["TinsUP"].append(STPH)
			dictionary[zone]["TinsDOWN"].append(STPL)
			dictionary[zone]["Costs"].append(cost)
			dictionary[zone]["Prices"].append(price)
			dictionary[zone]["Discomforts"].append(discomfort)
			dictionary[zone]["Real_Setpoints_High"].append(data[zone]["STPH"][i])
			dictionary[zone]["Real_Setpoints_Low"].append(data[zone]["STPL"][i])

	for zone in zones:
		with open(zone + '_' + string_extension + '.pckl', 'wb') as fp:
			pickle.dump(dictionary[zone], fp)

with open("cost_config.yml", 'r') as ymlfile:
	cfg = yaml.load(ymlfile)

if cfg["Server"]:
	cli = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
else:
	cli = get_client()

zones = cfg["Zones"]

popts = cfg["popts"]
kwh = {}
for i in zones:
	with open("../Buildings/"+cfg["Building"]+"/ZoneConfigs/"+i+".yml", 'r') as zonefile:
		cfgz = yaml.load(zonefile)
	kwh[i] = [cfgz["Advise"]["Heating_Consumption"], cfgz["Advise"]["Cooling_Consumption"]]


uspac = pytz.timezone(cfg["Pytz_Timezone"])
now = uspac.localize(datetime.datetime.strptime(cfg["Start_Date"], '%Y-%m-%d %H:%M:%S'))
policy_array = Utils.policy_per_minute(cfg["Setpoint"], now)
prices_array = Utils.prices_per_minute(cfg["Pricing"][cfg["Pricing"]["Energy_Rates"]], now, cfg["Pricing"]["DRs"])
policy_array_expanded = Utils.policy_per_minute_expanded(cfg["Setpoint"], now, cfg["Pricing"]["DRs"], cfg["Pricing"]["DRs_Expand_Percent"])


data = Utils.data_fetch(cfg, cli, zones)


"""
data = Utils.data_fetch(cfg, cli)
with open('temp.pckl', 'wb') as fp:
	pickle.dump(data, fp)
"""

baseline(data, policy_array, prices_array, zones, kwh, 'baseline')
baseline(data, policy_array_expanded, prices_array, zones, kwh, 'baseline_extended')
reality(data, policy_array, prices_array, zones, kwh, 'reality')
relive(data, policy_array, prices_array, zones, kwh, 'reality_simulated')
print "Finished"
