from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe
from xbos.services.hod import HodClient
import datetime
import pandas as pd
import pytz
from datetime import timedelta
import numpy as np
import math

def mins_in_day(timestamp):
	return timestamp.hour * 60 + timestamp.minute

def hamming_distance(a, b):
	return np.count_nonzero(a != b)

def find_similar_days(training_data, now, observation_length, k, method=hamming_distance):
	min_time = training_data.index[0] + timedelta(minutes=observation_length)
	# Find moments in our dataset that have the same hour/minute and is_weekend() == weekend.
	#print min_time
	selector = ((training_data.index.minute == now.minute) &
				(training_data.index.hour == now.hour) &
				(training_data.index > min_time))

	similar_moments = training_data[selector][:-1]
	obs_td = timedelta(minutes=observation_length)

	similar_moments['Similarity'] = [
		method(
			training_data[(training_data.index >= now - obs_td) &
							(training_data.index <= now)].get_values(),
			training_data[(training_data.index >= i - obs_td) &
							(training_data.index <= i)].get_values()
		) for i in similar_moments.index
		]

	indexes = (similar_moments.sort_values('Similarity', ascending=True).head(k).index)
	return indexes

def predict(data, now, similar_moments, prediction_time, resample_time):

	prediction = np.zeros((int(math.ceil(prediction_time/resample_time)) + 1, len(data.columns)))
	#print data[(data.index >= similar_moments[0]) & (data.index <= similar_moments[0] + timedelta(minutes=prediction_time))]
	for i in similar_moments:
		#print float(1. / float(len(similar_moments))) 
		#print data[(data.index >= i) & (data.index <= i + timedelta(minutes=prediction_time))]
		prediction += float(1.0 / float(len(similar_moments))) * data[(data.index >= i) & (data.index <= i + timedelta(minutes=prediction_time))]

	#print data[data.index == now],prediction[0]
	prediction[0] = data[data.index == now]
	time_index = pd.date_range(now, now+timedelta(minutes=prediction_time),freq='15T')
	return pd.DataFrame(data=prediction, index=time_index)



class Occupation:
	def __init__(self, now = datetime.datetime.now().replace(tzinfo=pytz.timezone("America/Los_Angeles"))):

		#c = get_client(agent = '172.17.0.1:28589', entity="thanos.ent")
		c = get_client()
		archiver = DataClient(c)
		hod = HodClient("ciee/hod",c)

		occ_query = """SELECT ?x ?room ?uuid WHERE {
			?x rdf:type/rdfs:subClassOf* brick:Occupancy_Sensor .
			?x bf:isLocatedIn ?room .
			?room bf:isPartOf bldg:SouthZone .
			?x bf:uuid ?uuid .
		};
		"""
		results = hod.do_query(occ_query)
		uuids = [x['?uuid'] for x in results['Rows']]

		temp_now = now

		start = '"' + (temp_now + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
		end = '"' + (temp_now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'

		dfs = make_dataframe(archiver.window_uuids(uuids, end, start, '15min', timeout=120))

		for uid, df in dfs.items():
			if 'mean' in df.columns:
				df = df[['mean']]
			df.columns = ['occ']
			dfs[uid] = df.resample('15min').mean()
			
		df = dfs.values()[0]
		if len(dfs) > 1:
			for newdf in dfs.values()[1:]:
				df['occ'] += newdf['occ']
		df['occ'] = 1*(df['occ']>0)

		observation_length_addition = 4*60
		k = 5
		prediction_time = 4*60
		resample_time = 15
		now = df.index[-prediction_time/resample_time]
		observation_length = mins_in_day(now) + observation_length_addition
		similar_moments = find_similar_days(df, now, observation_length, k)
		self.predictions = predict(df, now, similar_moments, prediction_time, resample_time)

	def occ(self, now_time):
		return self.predictions[0][now_time]
