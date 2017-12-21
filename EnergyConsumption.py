import datetime, pytz
from datetime import timedelta
from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe

class EnergyConsumption:
	def __init__(self, now=datetime.datetime.now(pytz.timezone("America/Los_Angeles")), heat=4000, cool=4000, vent=500):
		self.heat = heat
		self.cool = cool
		self.vent = vent

		#c = get_client(agent = '172.17.0.1:28589', entity="thanos.ent")
		c = get_client()
		archiver = DataClient(c)

		energy_cost= "9dc5b5cd-8cb1-3dd3-b582-5ed6bf3f0083"

		uuids = [energy_cost]

		start = '"' + (now + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
		end = '"' + (now - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') + ' PST"'
		
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				
		dfs = make_dataframe(archiver.window_uuids(uuids, end, start, '15min', timeout=120))

		for uid, df in dfs.items():
			
			if uid == uuids[0]:
				if 'mean' in df.columns:
					df = df[['mean']]
				df.columns = ['cost']
				
			dfs[uid] = df.resample('15min').mean()

		uid, self.df = dfs.items()[0]

	def calc_cost(self, action, time, period=15):
		if action == 'Heating' or action == '2':
			return (self.heat / period * 60 / 1000)*self.df['cost'][time]
		elif action == 'Cooling' or action == '1':
			return self.cool / period * 60 / 1000*self.df['cost'][time]
		elif action == 'Ventilation':
			return self.vent / period * 60 / 1000*self.df['cost'][time]
		elif action == 'Do Nothing' or action == '0':
			return 0
		else:
			print("picked wrong action")
			return 0
