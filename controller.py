import schedule
import datetime
import time
import math
import pytz
import pandas as pd

from Advise import Advise

from xbos import get_client
from xbos.services.hod import HodClientHTTP
from xbos.devices.thermostat import Thermostat
from xbos.services.pundat import DataClient, make_dataframe

#client = get_client(agent = '172.17.0.1:28589', entity="thanos.ent")
client = get_client()
hc = HodClientHTTP("http://ciee.cal-sdb.org")

q = """SELECT ?uri ?zone WHERE {
	?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
	?tstat bf:uri ?uri .
	?tstat bf:controls/bf:feeds ?zone .
};
"""

filename = "thermostat_changes.txt"

zones = {}
for tstat in hc.do_query(q):
	print tstat
	zones[tstat["?zone"]] = Thermostat(client, tstat["?uri"])

normal_zones = ["EastZone"]

def workday():
	p = {"override": True, "heating_setpoint": 70., "cooling_setpoint": 76.}
	print "workday",datetime.datetime.now()
	for z in normal_zones:
		print z,p
		zones[z].write(p)
def workday_inactive():
	p = {"override": True, "heating_setpoint": 62., "cooling_setpoint": 85.}
	print "workday inactive",datetime.datetime.now()
	for z in normal_zones:
		print z,p
		zones[z].write(p)

def normal_schedule():
	schedule.every().monday.at("08:00").do(workday)
	schedule.every().tuesday.at("08:00").do(workday)
	schedule.every().wednesday.at("08:00").do(workday)
	schedule.every().thursday.at("08:00").do(workday)
	schedule.every().friday.at("08:00").do(workday)

	schedule.every().monday.at("18:00").do(workday_inactive)
	schedule.every().tuesday.at("18:00").do(workday_inactive)
	schedule.every().wednesday.at("18:00").do(workday_inactive)
	schedule.every().thursday.at("18:00").do(workday_inactive)
	schedule.every().friday.at("18:00").do(workday_inactive)


	schedule.every().saturday.at("06:00").do(workday_inactive)
	schedule.every().sunday.at("06:00").do(workday_inactive)

	weekno = datetime.datetime.today().replace(tzinfo=pytz.timezone("America/Los_Angeles")).weekday()

	if weekno<5:
		now = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
		now_time = now.time()

		if now_time >= datetime.time(18,0) or now_time < datetime.time(7,0):
			workday_inactive()
		else:
			workday()
	else:
		workday_inactive()

	while True:
		schedule.run_pending()
		time.sleep(30)

def hvac_control(): 
	#try:

		#c = get_client(agent = '172.17.0.1:28589', entity="thanos.ent")
		c = get_client()
		archiver = DataClient(c)

		conf_set_high = "d38446d4-32cc-34bd-b293-0a3871a6759b"
		conf_set_low = "e4d39723-5907-35bd-a9b2-fc57b58b3779"

		uuids = [conf_set_high, conf_set_low]

		temp_now = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))

		start = '"' + temp_now.strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
		end = '"' + (temp_now - datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
		

		dfs = make_dataframe(archiver.window_uuids(uuids, end, start, '1min', timeout=120))

		for uid, df in dfs.items():
			
			if uid == uuids[0]:
				if 'mean' in df.columns:
					df = df[['mean']]
				df.columns = ['t_high']
			elif uid == uuids[1]:
				if 'mean' in df.columns:
					df = df[['mean']]
				df.columns = ['t_low']
			dfs[uid] = df.resample('1min').mean()
				
		df = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)
		f = open(filename, 'a')
		f.write("Did read: " + str(df['t_low'][-1]) + ", " + str(df['t_high'][-1]) + "\n")
		f.close()
				
		weekno = temp_now.weekday()

		if weekno<5:
			now = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
			now_time = now.time()

			if now_time >= datetime.time(18,0) or now_time < datetime.time(7,0):
				heating_setpoint = 62.
				cooling_setpoint = 85.
			else:
				heating_setpoint = 70.
				cooling_setpoint = 76.
		else:
			heating_setpoint = 62.
			cooling_setpoint = 85.
		
		adv = Advise()
		action, temp = adv.advise()
		temp = float(temp)
		if action == "0":
			p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": cooling_setpoint}
			f = open(filename, 'a')
			f.write("Did write: " + str(heating_setpoint) + ", " + str(cooling_setpoint) + "\n")
			f.close()
			for z in normal_zones:
				print p
				#zones[z].write(p)
		elif action == "1":
			p = {"override": True, "heating_setpoint": heating_setpoint, "cooling_setpoint": math.floor(temp-0.1)}
			f = open(filename, 'a')
			f.write("Did write: " + str(heating_setpoint) + ", " + str(math.floor(temp-0.1)) + "\n")
			f.close()
			for z in normal_zones:
				print p
				#zones[z].write(p)
		elif action == "2":
			p = {"override": True, "heating_setpoint": math.ceil(temp+0.1), "cooling_setpoint": cooling_setpoint}
			f = open(filename, 'a')
			f.write("Did write: " + str(math.ceil(temp+0.1)) + ", " + str(cooling_setpoint) + "\n")
			f.close()
			for z in normal_zones:
				print p
				#zones[z].write(p)
		else:
			print("PROBLEM")
			#normal_schedule()
	#except:
	#	normal_schedule()

if __name__ == '__main__':
	f = open(filename   , 'w')
	f.close()
	starttime=time.time()
	while True:
		hvac_control()
		time.sleep(60.*15. - ((time.time() - starttime) % (60.*15.)))