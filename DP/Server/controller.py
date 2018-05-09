import datetime, time, math, pytz, os, sys, threading
import pandas as pd
import yaml
from NormalSchedule import NormalSchedule
from DataManager import DataManager
from Advise import Advise
from xbos import get_client
from xbos.services.hod import HodClientHTTP
from xbos.services.hod import HodClient
from xbos.devices.thermostat import Thermostat

# TODO only one zone at a time, making multizone comes soon

filename = "thermostat_changes.txt"  # file in which the thermostat changes are recorded

# the main controller
def hvac_control(cfg, advise_cfg, tstat):

	now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))

	dataManager = DataManager(cfg, advise_cfg, now=now)

	t_high, t_low, t_mode = dataManager.thermostat_setpoints()
	# document the "before" state
	try:
		f = open(filename, 'a')
		f.write("Did read: " + str(t_low) + ", " + str(t_high) + ", " + str(t_mode) + "\n")
		f.close()
	except:
		print "Could not document changes."

	try:
		Prep_Therm = dataManager.preprocess_therm()
		setpoints_array = dataManager.building_setpoints()
		adv = Advise(now.astimezone(tz=pytz.timezone(cfg["Pytz_Timezone"])),
					 dataManager.preprocess_occ(),
					 Prep_Therm,
					 dataManager.weather_fetch(),
					 dataManager.prices(),
					 advise_cfg["Advise"]["Lambda"],
					 cfg["Interval_Length"],
					 advise_cfg["Advise"]["Hours"],
					 advise_cfg["Advise"]["Print_Graph"],
					 advise_cfg["Advise"]["Maximum_Safety_Temp"],
					 advise_cfg["Advise"]["Minimum_Safety_Temp"],
					 advise_cfg["Advise"]["Heating_Consumption"],
					 advise_cfg["Advise"]["Cooling_Consumption"],
					 advise_cfg["Advise"]["Max_Actions"],
					 advise_cfg["Advise"]["Thermal_Precision"],
					 advise_cfg["Advise"]["Occupancy_Obs_Len_Addition"],
					 setpoints_array)
		action = adv.advise()
		temp = float(Prep_Therm['t_next'][-1])
	except:
		e = sys.exc_info()[0]
		print e
		return False


	heating_setpoint = setpoints_array[0][0]
	cooling_setpoint = setpoints_array[0][1]
	# action "0" is Do Nothing, action "1" is Cooling, action "2" is Heating
	if action == "0":
		p = {"override": True, "heating_setpoint": math.floor(temp-0.1)-1, "cooling_setpoint": math.ceil(temp+0.1)+1, "mode": 3}
		print "Doing nothing"
		print p

		# document changes
		try:
			f = open(filename, 'a')
			f.write("Did write: " + str(math.floor(temp-0.1)-1) + ", " + str(math.ceil(temp+0.1)+1) + ", " + str(3) +"\n")
			f.close()
		except:
			print "Could not document changes."
			
	elif action == "1":
		p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": math.floor(temp-0.1), "mode": 3}
		print "Heating"
		print p

		# document changes
		try:
			f = open(filename, 'a')
			f.write("Did write: " + str(heating_setpoint) + ", " + str(math.floor(temp-0.1)) + ", " + str(3) + "\n")
			f.close()
		except:
			print "Could not document changes."
		
	elif action == "2":
		p = {"override": True, "heating_setpoint": math.ceil(temp+0.1), "cooling_setpoint": cooling_setpoint, "mode": 3}
		print "Cooling"
		print p

		# document changes
		try:
			f = open(filename, 'a')
			f.write("Did write: " + str(math.ceil(temp+0.1)) + ", " + str(cooling_setpoint) + ", " + str(3) + "\n")
			f.close()
		except:
			print "Could not document changes."
	else:
		print "Problem with action."
		return False

	# try to commit the changes to the thermostat, if it doesnt work 10 times in a row ignore and try again later

	for i in range(cfg["Thermostat_Write_Tries"]):
		try:
			tstat.write(p)
			break
		except:
			if i == cfg["Thermostat_Write_Tries"] - 1:
				e = sys.exc_info()[0]
				print e
				return False
			continue

	return True

class ZoneThread (threading.Thread):

	def __init__(self, cfg, tstat, zone):
		threading.Thread.__init__(self)
		self.cfg = cfg
		self.tstat = tstat
		self.zone = zone

	def run(self):

		try:
			with open('ZoneConfigs/'+self.zone+'.yml', 'r') as ymlfile:
				advise_cfg = yaml.load(ymlfile)

			print advise_cfg
			if not hvac_control(self.cfg, advise_cfg, self.tstat):
				print("Problem with MPC, entering normal schedule.")
				normal_schedule = NormalSchedule(cfg, tstat)
				normal_schedule.normal_schedule()
		except:
			normal_schedule = NormalSchedule(cfg, tstat)
			normal_schedule.normal_schedule()

if __name__ == '__main__':

	starttime = time.time()
	while True:
		# read from config file
		try:
			yaml_filename = sys.argv[1]
		except:
			sys.exit("Please specify the configuration file as: python2 controller.py config_file.yaml")

		with open(yaml_filename, 'r') as ymlfile:
			cfg = yaml.load(ymlfile)

		if not os.path.exists(filename):
			f = open(filename   , 'w')
			f.close()

		if cfg["Server"]:
			client = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
		else:
			client = get_client()
		hc = HodClient(cfg["Hod_Client"], client)

		q = """SELECT ?uri ?zone WHERE {
			?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
			?tstat bf:uri ?uri .
			?tstat bf:controls/bf:feeds ?zone .
		};
		"""

		threads = []
		for tstat in hc.do_query(q)['Rows']:
			print tstat
			thread = ZoneThread(cfg, Thermostat(client, tstat["?uri"]), tstat["?zone"])
			thread.start()
			threads.append(thread)

		for t in threads:
			t.join()

		print datetime.datetime.now()
		time.sleep(60.*15. - ((time.time() - starttime) % (60.*15.)))
