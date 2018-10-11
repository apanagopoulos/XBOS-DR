import datetime, pytz
import pickle
from xbos import get_client
from xbos.services.hod import HodClient
from xbos.services import mdal
from datetime import timedelta
import pandas as pd
import yaml


def in_between(now, start, end):
	if start < end:
		return start <= now < end
	elif end < start:
		return start <= now or now < end
	else:
		return True

def data_fetch(cfg, cli, zones):
	# state, tin, tout, setp_high, setp_low
	UUIDS = {"SouthZone": ["dfb2b403-fd08-3e9b-bf3f-18c699ce40d6", "03099008-5224-3b61-b07e-eee445e64620",
						   "1c467b79-b314-3c1e-83e6-ea5e7048c37b", "dbbf4a91-107a-3b15-b2c0-a49b54116daa",
						   "eeadc8ed-6255-320d-b845-84f44748fe95"],
			 "NorthZone": ["5e55e5b1-007b-39fa-98b6-ae01baa6dccd", "c7e33fa6-f683-36e9-b97a-7f096e4b57d4",
						   "1c467b79-b314-3c1e-83e6-ea5e7048c37b", "9fa56ac1-0f8a-3ad2-86e8-72e816b875ad",
						   "e4e0db0b-1c15-330e-a864-011e558f542e"],
			 "CentralZone": ["187ed9b8-ee9b-3042-875e-088a08da37ae", "c05385e5-a947-37a3-902e-f6ea45a43fe8",
							 "1c467b79-b314-3c1e-83e6-ea5e7048c37b", "0d037818-02c2-3e5b-87e9-94570d43b418",
							 "6cbee2ae-06e7-3fc3-a2fc-698fa3deadee"],
			 "EastZone": ["7e543d07-16d1-32bb-94af-95a01f4675f9", "b47ba370-bceb-39cf-9552-d1225d910039",
						  "1c467b79-b314-3c1e-83e6-ea5e7048c37b", "d38446d4-32cc-34bd-b293-0a3871a6759b",
						  "e4d39723-5907-35bd-a9b2-fc57b58b3779"]}

	uspac = pytz.timezone(cfg["Pytz_Timezone"])
	startime = uspac.localize(datetime.datetime.strptime(cfg["Start_Date"], '%Y-%m-%d %H:%M:%S')).astimezone(
		tz=pytz.utc)
	endtime = startime + timedelta(hours=24)

	hod = HodClient(cfg["Building"]+"/hod", cli)

	occ_query = """SELECT ?sensor ?uuid ?zone WHERE {
				  ?sensor rdf:type brick:Occupancy_Sensor .
				  ?sensor bf:isLocatedIn/bf:isPartOf ?zone .
				  ?sensor bf:uuid ?uuid .
				  ?zone rdf:type brick:HVAC_Zone
				};
				"""  # get all the occupancy sensors uuids

	results = hod.do_query(occ_query)  # run the query
	occ_uuids = [[x['?zone'], x['?uuid']] for x in results['Rows']]  # unpack


	dataframes = {}

	for zone in zones:

		query_list = []
		for i in occ_uuids:
			if i[0] == zone:
				query_list.append(i[1])


		c = mdal.MDALClient("xbos/mdal", client=cli)
		dfs = c.do_query({'Composition': UUIDS[zone],
						  'Selectors': [mdal.MAX, mdal.MEAN, mdal.MEAN, mdal.MEAN, mdal.MEAN],
						  'Time': {'T0': startime.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
								   'T1': endtime.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
								   'WindowSize': '1min',
								   'Aligned': True}})

		df = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)
		df_th = df.rename(
			columns={UUIDS[zone][0]: 'State', UUIDS[zone][1]: 'Tin', UUIDS[zone][2]: 'Tout', UUIDS[zone][3]: "STPH",
					 UUIDS[zone][4]: "STPL"})
		df_th['change_of_action'] = (df_th['State'].diff(1) != 0).astype('int').cumsum()

		c = mdal.MDALClient("xbos/mdal", client=cli)
		dfs = c.do_query({'Composition': query_list,
						  'Selectors': [mdal.MAX] * len(query_list),
						  'Time': {'T0': startime.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
								   'T1': endtime.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
								   'WindowSize': '1min',
								   'Aligned': True}})

		dfs = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)

		df = dfs[[query_list[0]]]
		df.columns.values[0] = 'Occ'
		df.is_copy = False
		df.columns = ['Occ']
		# perform OR on the data, if one sensor is activated, the whole zone is considered occupied
		for i in range(1, len(query_list)):
			df.loc[:, 'Occ'] += dfs[query_list[i]]
		df.loc[:, 'Occ'] = 1 * (df['Occ'] > 0)
		df_occ = df

		dataframes[zone] = pd.concat([df_th, df_occ], axis=1)
	return dataframes


def policy_per_minute(policy, now):

	now_time = now
	setpoints = []

	while now_time <= now + timedelta(hours=24):
		i = 0 if now_time.weekday() < 5 else 1
		for j in policy[i]:
			if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
						  datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
				setpoints.append([j[2], j[3]])
				break

		now_time += timedelta(minutes=1)

	return setpoints[:-1]

def policy_per_minute_expanded(policy, now, DRs, expand_percent):

	now_time = now
	setpoints = []

	while now_time <= now + timedelta(hours=24):
		i = 0 if now_time.weekday() < 5 else 1

		flag = False
		for dr in DRs:
			if in_between(now_time.time(), datetime.time(int(dr[1].split(":")[0]), int(dr[1].split(":")[1])),
						  datetime.time(int(dr[2].split(":")[0]), int(dr[2].split(":")[1]))) and \
					(now_time.date() == datetime.datetime.strptime(dr[0], "%Y-%m-%d").date()):
				flag = True
				break



		for j in policy[i]:

			if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
						  datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
				if flag:
					setpoints.append([j[2] - expand_percent / 100. * j[2], j[3] + expand_percent / 100. * j[3]])
				else:
					setpoints.append([j[2], j[3]])
				break

		now_time += timedelta(minutes=1)
	return setpoints[:-1]


def prices_per_minute(price_array, now, DRs):
	now_time = now
	pricing = []

	while now_time <= now + timedelta(hours=24):
		i = 0 if now_time.weekday() < 5 else 1

		flag = True
		for dr in DRs:
			if in_between(now_time.time(), datetime.time(int(dr[1].split(":")[0]), int(dr[1].split(":")[1])),
						  datetime.time(int(dr[2].split(":")[0]), int(dr[2].split(":")[1]))) and \
					(now_time.date() == datetime.datetime.strptime(dr[0], "%Y-%m-%d").date()):
				pricing.append(dr[3])
				flag = False
				break

		for j in price_array[i]:

			if not flag:
				break
			if in_between(now_time.time(), datetime.time(int(j[0].split(":")[0]), int(j[0].split(":")[1])),
						  datetime.time(int(j[1].split(":")[0]), int(j[1].split(":")[1]))):
				pricing.append(j[2])
				break

		now_time += timedelta(minutes=1)

	return pricing[:-1]

def discomfort(STPH, STPL, Tin, Occ):
	if abs(STPL - Tin) < abs(STPH - Tin):
		discomfort = (STPL - Tin) ** 2.
	else:
		discomfort = (STPH - Tin) ** 2.
	# return 0 if inside setpoints, discomfort*occupancy-probability else
	if Tin > STPL and Tin < STPH:
		return 0
	else:
		return discomfort * Occ

def action(STPH, STPL, Tin):
	hysterisis = 1
	if Tin < STPL-hysterisis:
		return 1.
	elif Tin > STPH+hysterisis:
		return 2.
	else:
		return 0.

def cost(action, price, heatE, coolE):

	if action == 'Cooling' or action == 2.:
		return (coolE / 60.) * price
	elif action == 'Heating' or action == 1.:
		return (heatE / 60.) * price
	else:
		return 0


def T_next(X, c1, c2, c3, c4, c5, c6, c7):

		Tin, a, Tout, dt, zones = X

		zone1 = zones[0]
		zone2 = zones[1]
		zone3 = zones[2]

		if a == 0. or a == 3.:
			a1, a2 = 0, 0
		elif a == 1.:
			a1, a2 = 1, 0
		else:
			a1, a2 = 0, 1

		return Tin + (c1 * a1 * Tin + c2 * a2 * Tin + c3 * (Tout - Tin) + c4 + c5 * (Tin - zone1) +
					  c6 * (Tin - zone2) + c7 * (Tin - zone3)) * dt

