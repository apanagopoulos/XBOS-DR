import datetime, pytz
from xbos import get_client
from xbos.services import mdal
import pandas as pd
import yaml


UUIDS = {"SouthZone":"dfb2b403-fd08-3e9b-bf3f-18c699ce40d6", "NorthZone":"5e55e5b1-007b-39fa-98b6-ae01baa6dccd",
		 "CentralZone":"187ed9b8-ee9b-3042-875e-088a08da37ae", "EastZone":"7e543d07-16d1-32bb-94af-95a01f4675f9"}

def preprocess_data(UUID, c, startime, endtime):

	c = mdal.MDALClient("xbos/mdal", client=c)
	dfs = c.do_query({'Composition': [UUID],
					  'Selectors': [mdal.MAX],
					  'Time': {'T0': startime.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
							   'T1': endtime.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
							   'WindowSize': '1min',
							   'Aligned': True}})


	df = [dframe for uid, dframe in dfs.items()][0]
	df = df.rename(columns={UUID: 'State'})
	df = df.loc[df['State'] != 0.0]
	return df.dropna()

def cost_calculator(df, DRs, price_array):

	def in_between(now, start, end):
		if start < end:
			return start <= now < end
		elif end < start:
			return start <= now or now < end
		else:
			return True

	def kwH(action, heat=0.075, cool=1.25, vent=0.02):
		if action == 2. :
			return heat / 60.
		elif action == 1 :
			return cool / 60.
		else:
			return vent / 60.

	pricing = []
	for index, row in df.iterrows():

		i = 1 if index.weekday() >= 5  else 0

		flag = True
		for dr in DRs:
			if in_between(index.time(), datetime.time(int(dr[1].split(":")[0]), int(dr[1].split(":")[1])),
						  datetime.time(int(dr[2].split(":")[0]), int(dr[2].split(":")[1]))) and \
					(index.date() == datetime.datetime.strptime(dr[0], "%Y-%m-%d").date()):
				pricing.append([index, dr[3] * kwH(row["State"])])
				flag = False
				break

		for j in price_array[i]:

			if not flag:
				break

			if in_between(index.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
								   datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))) :

				pricing.append([index,j[2]*kwH(row["State"])])
				break

	return pricing



with open("cost_config.yml", 'r') as ymlfile:
	cfg = yaml.load(ymlfile)

if cfg["Server"]:
	c = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
else:
	c = get_client()

uspac = pytz.timezone(cfg["Pytz_Timezone"])
startime = uspac.localize(datetime.datetime.strptime(cfg["Start_Date"], '%Y-%m-%d %H:%M:%S')).astimezone(tz=pytz.utc)
endtime = uspac.localize(datetime.datetime.strptime(cfg["End_Date"], '%Y-%m-%d %H:%M:%S')).astimezone(tz=pytz.utc)

for i in cfg["Zones"]:
	df = preprocess_data(UUIDS[i], c, startime, endtime)

	price_array = cfg["Pricing"][cfg["Pricing"]["Energy_Rates"]]
	DRs = cfg["Pricing"]["DRs"]
	costs = cost_calculator(df, DRs, price_array)
	print i + " : " + str(sum([i[1] for i in costs])) + " USD"