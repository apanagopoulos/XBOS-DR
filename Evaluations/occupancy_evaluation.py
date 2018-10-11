from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe
from xbos.services.hod import HodClient
import datetime
import pandas as pd
import pytz
from datetime import timedelta
import numpy as np
import math
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import mean_absolute_error

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

	indexes = (similar_moments.sort_values('Similarity', ascending=True)
				.head(k).index)
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


zones = ['SouthZone', 'EastZone', 'CentralZone', 'NorthZone']
temp_now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(tz=pytz.timezone("America/Los_Angeles"))
c = get_client()
archiver = DataClient(c)
hod = HodClient("ciee/hod",c)

occ_query = """SELECT ?sensor ?uuid ?zone WHERE {
  ?sensor rdf:type brick:Occupancy_Sensor .
  ?sensor bf:isLocatedIn/bf:isPartOf ?zone .
  ?sensor bf:uuid ?uuid .
  ?zone rdf:type brick:HVAC_Zone
};
"""

results = hod.do_query(occ_query)
uuids = [[x['?zone'],x['?uuid']] for x in results['Rows']]

per_zone_occ_list = []

for zone_name in zones:

	query_list = []
	for i in uuids:
		if i[0] == zone_name:
			query_list.append(i[1])

	start = '"' + temp_now.strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
	end = '"' + (temp_now - timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'

	dfs = make_dataframe(archiver.window_uuids(query_list, end, start, '15min', timeout=120))

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

	df.index = df.index.tz_localize(pytz.timezone("America/Los_Angeles"))
	per_zone_occ_list.append(df)

observation_length_addition = 4*60
k = 5
prediction_hours = 8
prediction_time = prediction_hours*60
resample_time = 15 # prediction_time mod resample_time must be 0
days_of_data = 30

sticks = []
disc= 4
for i in range(0, 4*prediction_hours+1, disc):
	if int(round(i/4))<10:
		hours = "0"+str(int(round(i/4)))
	else:
		hours = str(int(round(i/4)))
	if int(round(i%4))<10:
		minutes = "0"+str(int(round(i%4)))
	else:
		minutes = str(int(round(i%4)))
	sticks.append(hours+":"+minutes)

pos = np.arange(4*prediction_hours+1)
ind = np.arange(0,4*prediction_hours+1)
fig, (ax1,ax2) = plt.subplots(2, sharex=True)
ax1.set_xticks(pos[::disc])
ax1.set_xticklabels(sticks)
ax1.set_xlim(0, 4*prediction_hours)
ax1.set_ylabel(r'RMSE (Probability of occupancy)')
#ax1.legend(loc=4, ncol=3, numpoints=1, fancybox=False, framealpha=0.2)

ax2.set_ylabel(r'Missrate (%)')
#ax2.legend(loc=4, ncol=3, numpoints=1, fancybox=False, framealpha=0.2)

count = 0
for df in per_zone_occ_list:
	print "Preparing next zone..."
	temp_now = df.index[-1]
	prediction_list = []
	ground_truth_list = []

	hamming_distance = [0]*(prediction_hours*4+1)

	while temp_now > df.index[0] + timedelta(days=days_of_data):

		temp = df.ix[temp_now - timedelta(days=days_of_data):temp_now]
		
		now = temp.index[-(prediction_time/resample_time+1)]
		observation_length = mins_in_day(now) + observation_length_addition
		similar_moments = find_similar_days(temp, now, observation_length, k)
		prediction = predict(temp, now, similar_moments, prediction_time, resample_time)[0]
		ground_truth = temp[-(1+prediction_time/resample_time):]['occ'].tolist()
		prediction_list.append(prediction)
		ground_truth_list.append(ground_truth)
		
		for i in range(prediction_hours*4+1):
			if prediction[i]>=0.5:
				hamming_distance[i] += 1-ground_truth[i]
			else:
				hamming_distance[i] += ground_truth[i]

		temp_now = temp_now - timedelta(hours=randint(1,2))


	hamming_distance = [x *100 for x in hamming_distance]
	hamming_distance = [x /len(prediction_list) for x in hamming_distance]

	prediction_tin = [i for i in prediction_list]
	groundTruth_tin = [i for i in ground_truth_list]
	rmse = np.sqrt(np.mean(np.square(np.subtract(prediction_tin,groundTruth_tin)),axis=0))
	#mae = mean_absolute_error(prediction_tin, groundTruth_tin,multioutput='raw_values')
	index = np.arange(4*8+1)
	ax1.plot(index,rmse,'-', label = zones[count]+' RMSE')
	ax2.plot(index,hamming_distance,'-',label=zones[count]+' missrate')
	count += 1

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
#handles, labels = ax2.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)

plt.xlabel('Predictive horizon (Hours)')
plt.legend()
plt.show()
