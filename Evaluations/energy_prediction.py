from __future__ import division
import datetime, pytz
from datetime import timedelta
from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe
from house import IEC

import numpy as np
import pandas as pd


# from matplotlib.pyplot import step, xlim, ylim, show
import matplotlib.pyplot as plt

c = get_client()
archiver = DataClient(c)

demand= "4d6e251a-48e1-3bc0-907d-7d5440c34bb9"

uuids = [demand]
now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(tz=pytz.timezone("America/Los_Angeles"))

start = '"' + now.strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
end = '"2017-9-21 00:00:00 PST"'


dfs = make_dataframe(archiver.window_uuids(uuids, end, start, '1min', timeout=120))

for uid, df in dfs.items():

	if uid == uuids[0]:
		if 'mean' in df.columns:
			df = df[['mean']]
		df.columns = ['House Consumption']

	dfs[uid] = df.resample('1min').mean()

uid, df = dfs.items()[0]

df = df/(1000*60)

print df.resample('60T').sum().describe()

"""
data = pd.DataFrame()
data.index = pd.DatetimeIndex(pd.datarange(HISTORICAL RANGE))
data['House Consumption'] = Kwh consumption (ANA LEPTO)
data = data.interpolate(max_time= '15T')
"""

df = df.tz_localize('UTC').tz_convert(pytz.timezone("America/Los_Angeles"))

yesterday = now - datetime.timedelta(hours=12)

prediction = IEC(df[:yesterday], prediction_window=12*60).predict(["Baseline Finder"])
#prices.index = data.index
#prices['US'] = Dollar/Kwh

index = np.arange(12*60)
plt.plot(index,prediction[["Baseline Finder"]], label="Energy Prediction")
plt.plot(index,df[["House Consumption"]][-12*60:], label="Ground Truth")
plt.xlabel('Predictive horizon (Minutes)')
plt.ylabel(r'KWh')
plt.legend()
plt.show()
#data * prices = posa plirwses
