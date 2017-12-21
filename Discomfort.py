import datetime, pytz
from datetime import timedelta

class Discomfort:
	def __init__(self, now=datetime.datetime.now(pytz.timezone("America/Los_Angeles"))):
		
		self.temp_now = now

	def disc(self, t_in, occ, node_time):

		weekno = self.temp_now.weekday()

		if weekno<5:
			now_time = (self.temp_now + timedelta(minutes=node_time)).time()

			if now_time >= datetime.time(18,0) or now_time < datetime.time(7,0):
				heating_setpoint = 62.
				cooling_setpoint = 85.
			else:
				heating_setpoint = 70.
				cooling_setpoint = 76.
		else:
			heating_setpoint = 62.
			cooling_setpoint = 85.

		# check which setpoint is the temperature closer to
		if abs(heating_setpoint - t_in) < abs(cooling_setpoint - t_in):
			discomfort = (heating_setpoint - t_in) ** 2.
		else:
			discomfort = (cooling_setpoint - t_in) ** 2.
		# return 0 if inside setpoints, discomfort*occupancy-probability else
		if t_in > heating_setpoint and t_in < cooling_setpoint:
			return 0
		else:
			return discomfort*occ