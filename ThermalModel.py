import datetime, pytz
import pandas as pd
from scipy.optimize import curve_fit
from datetime import timedelta
from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe

def func(X, c1, c2, c3, c4):
	Tin, a1, a2 = X
	return c1 * a1 * Tin + c2 * a2 * Tin + c3 * Tin + c4

class ThermalModel:

	def __init__(self, now=datetime.datetime.now(pytz.timezone("America/Los_Angeles"))):

		self.current_time = now
		self.df = self.preprocess_thermal()
		self.popt, pcov = curve_fit(func, self.df[['tin', 'a1' , 'a2']].T.as_matrix(), self.df['temp_next'].as_matrix())

	def next_temperature(self, Tin, action, zone=0):
		if action == 'Heating' or action == '2':
			return round(func([Tin, 1, 0], *self.popt) * 400) / 400.0
		elif action == 'Cooling' or action == '1':
			return round(func([Tin, 0, 1], *self.popt) * 400) / 400.0
		else:
			return round(func([Tin, 0, 0], *self.popt) * 400) / 400.0

	def preprocess_thermal(self):

		def f1(row):
			if row['a'] > 0 and row['a']<=1:
				val = 1
			else:
				val = 0
			return val

		def f2(row):
			if row['a']>1:
				val = 1
			else:
				val = 0
			return val

		#c = get_client(agent = '172.17.0.1:28589', entity="thanos.ent")
		c = get_client()
		archiver = DataClient(c)

		se_temp = "b47ba370-bceb-39cf-9552-d1225d910039"
		se_state = "7e543d07-16d1-32bb-94af-95a01f4675f9"

		uuids = [se_temp, se_state]

		temp_now = self.current_time

		start = '"' + (temp_now + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
		end = '"' + (temp_now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'

		dfs = make_dataframe(archiver.window_uuids(uuids, end, start, '15min', timeout=120))

		for uid, df in dfs.items():
			
			if uid == uuids[0]:
				if 'mean' in df.columns:
					df = df[['mean']]
				df.columns = ['tin']
			elif uid == uuids[1]:
				if 'max' in df.columns:
					df = df[['max']]
				df.columns = ['a']
				
			dfs[uid] = df.resample('15min').mean()

		df = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)
		df['a1'] = df.apply(f1, axis=1)
		df['a2'] = df.apply(f2, axis=1)
		df['tin']=df['tin'].replace(to_replace=0, method='pad')
		df['temp_next'] = df['tin'].shift(-1)

		return df.dropna()