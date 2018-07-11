import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.append("./../MPC")
sys.path.append("./..")

import utils

from xbos import get_client

from DataManager import DataManager

from Discomfort import Discomfort
from EnergyConsumption import EnergyConsumption
from Occupancy import Occupancy
from Safety import Safety
from ThermalDataManager import ThermalDataManager

import pytz
import os

import datetime


# --------------------------------------------------------------------------------------------

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def choosebuildingandzone():
	 print "-----------------------------------"
	 print "Buildings:"
	 print "-----------------------------------"
	 root, dirs, files = os.walk("../Buildings/").next()
	 for index, building in enumerate(dirs, start=1):
	 	 print index, building
	 print "-----------------------------------"
	 index = input("Please choose a building (give a number):") - 1
	 building = dirs[index]
	 print "-----------------------------------"
	 print ""
	 print "-----------------------------------"
	 print "	" + str(building)
	 print "-----------------------------------"
	 print "-----------------------------------"
	 print "Zones:"
	 print "-----------------------------------"
	 root, dirs, files = os.walk("../Buildings/" + str(building) + "/ZoneConfigs").next()
	 for index, zones in enumerate(files, start=1):
	 	 print index, zones[:-4]
	 print "-----------------------------------"
	 index = input("Please choose a zone (give a number):") - 1
	 zone = files[index][:-4]
	 print "-----------------------------------"
	 print "-----------------------------------"
	 print "	" + str(building)
	 print "	" + str(zone)
	 print "-----------------------------------"
	 return building, zone


def FtoC(x):
	 return x


# return (x-32)*5/9.

def PlotDay(OPs, Tins, Tout, Policy, TinsUP, TinsDOWN, TinsUP2, TinsDOWN2, TinsUP3, TinsDOWN3, Costs, Prices, Discomforts, method, manual, building, zone, date,):
	 '''
	 OPs : ground truth occupancy array (1440 binary values) 
	 Tins : 1440 F Indoor temperature values
	 Policy : 1440 0,1,2,3 values corresponding to nothing, heating, cooling, ventilation respectively
	 TinsUP : 1440 F Cooling setpoint temperature values
	 TinsDOWN : 1440 F Heating setpoint temperature values
	 Costs : 1440 dollars values
	 Prices : 1440 dollars values
	 Discomforts : 1440 F^2 min values of discomfortPolicies
	 method : any string describing the method
	 '''

	 discomfort = sum(Discomforts[:])
	 Costs = [sum(Costs[:i]) for i in range(1, len(Costs) + 1)]
	 Tins = [FtoC(i) for i in Tins]
	 Tout = [FtoC(i) for i in Tout]
	 TinsUP = [FtoC(i) for i in TinsUP]
	 TinsDOWN = [FtoC(i) for i in TinsDOWN]
	 TinsUP2 = [FtoC(i) for i in TinsUP2]
	 TinsDOWN2 = [FtoC(i) for i in TinsDOWN2]
	 TinsUP3 = [FtoC(i) for i in TinsUP3]
	 TinsDOWN3 = [FtoC(i) for i in TinsDOWN3]

	 Interval = 1
	 sticks = []
	 sticksDensity = 180 / Interval
	 for i in range(0, 60 * 24 / Interval, sticksDensity):
	 	 if int(round(i * Interval / 60)) < 10:
	 	 	 hours = "0" + str(int(round(i * Interval / 60)))
	 	 else:
	 	 	 hours = str(int(round(i * Interval / 60)))
	 	 if int(round(i * Interval % 60)) < 10:
	 	 	 minutes = "0" + str(int(round(i * Interval % 60)))
	 	 else:
	 	 	 minutes = str(int(round(i * Interval % 60)))
	 	 sticks.append(hours + ":" + minutes)

	 pos = np.arange(60 * 24 / Interval)
	 width = 1.0  # gives histogram aspect to the bar diagram

	 fig = plt.figure()

	 gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1])  #

	 ax = fig.add_subplot(gs[0])  #
	 ax.set_xticks(pos[::sticksDensity] + (width / 2))
	 ax.set_xticklabels(sticks)
	 ax.set_xlim([0, 24 * 60 / Interval])
	 ax.plot(pos, Tins[:], label="$T^{ IN}$", color='red')
	 ax.plot(pos, Tout[:], label="$T^{ OUT}$", color='cyan')
	 ax.plot(pos, TinsUP[:], label="$T^{ UP}$", color='blue')
	 ax.plot(pos, TinsDOWN[:], label="$T^{ DOWN}$", color='blue')
	 ax.plot(pos, TinsUP2[:], label="$T^{ UP}$", color='yellow')
	 ax.plot(pos, TinsDOWN2[:], label="$T^{ DOWN}$", color='yellow')
	 if manual:
	 	 for i in manual:
	 	 	 ax.annotate('manual', xy=(pos[i], TinsUP3[i]), xytext=(pos[i], TinsUP3[i]+5), arrowprops=dict(facecolor='red', shrink=0.05),)
	 	 	 ax.annotate('manual', xy=(pos[i], TinsDOWN3[i]), xytext=(pos[i], TinsDOWN3[i]-5), arrowprops=dict(facecolor='red', shrink=0.05),)
	 ax.plot(pos, TinsUP3[:], label="$T^{ UP}$", color='orange')
	 ax.plot(pos, TinsDOWN3[:], label="$T^{ DOWN}$", color='orange')

	 ax4 = ax.twinx()
	 ax4.plot(pos, Costs[:], color='green')
	 ax4.set_ylabel('Cost ($)')

	 ax.set_ylim(0.1, 2500000)

	 ax.set_ylabel(r"Temperature ($^\circ$F)")
	 # ax.set_ylim(0.1, 35) in C
	 ax.set_ylim(50.1, 110.1)
	 ax.xaxis.grid()
	 ax.yaxis.grid()

	 ax2 = ax.twinx()
	 ax2.set_xticks(pos[::sticksDensity] + (width / 2))
	 ax2.set_xticklabels(sticks)
	 ax2.set_xlim([0, 24 * 60 / Interval])
	 ax2.plot(0, 0, label="$T^{ IN}$", color='red')
	 ax2.plot(0, 0, label="$T^{ OUT}$", color='cyan')
	 ax2.plot(0, 0, label="Cost", color='green')
	 ax2.plot(0, 0, label="Comfort-band limits", color='blue')
	 ax2.plot(0, 0, label="Safety-band limits", color='yellow')
	 ax2.plot(0, 0, label="Setpoint limits", color='orange')
	 ax2.plot(0, 0, label="HVAC state", color='red')
	 ax2.plot(0, 0, label="Prices", color='purple')

	 ax2.bar(pos, OPs, width, color='grey', alpha=0.4, label="Occupancy", linewidth=0)
	 # ax2.bar(0, 0, 0, color='grey', alpha=0.7, label="Occupancy", linewidth=0)
	 ax2.legend(loc=2, ncol=6)
	 ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=9)  # , fancybox=True, shadow=True)
	 # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol64, fancybox=True, shadow=True)
	 # ax2.set_ylim(2, 27)
	 group_labels1 = ['']
	 ax2.set_yticklabels(group_labels1)
	 ax2.yaxis.set_ticks(np.arange(0, 1, 1))

	 ax3 = fig.add_subplot(gs[1], sharex=ax)  #
	 # ax3.set_xticks(pos[::sticksDensity] + (width / 2))
	 # ax3.set_xticklabels(sticks)
	 ax3.set_xlim([0, 24 * 60 / Interval])
	 ax3.set_ylabel('Action')
	 ax3.plot(pos, Policy[:], label="Policy of Function", color='red')
	 # xticklabels = ax.get_xticklabels()+ax2.get_xticklabels()
	 # plt.setp(xticklabels, visible=False)
	 plt.subplots_adjust(hspace=0.001)
	 # ax3.set_xlabel('Time')
	 ax3.set_ylim(-1, 6)
	 group_labels = ['Nothing', 'Heating I', 'Cooling I', 'Ventilation', 'Heating II', 'Cooling II',]
	 ax3.set_yticklabels(group_labels)
	 ax3.yaxis.grid()
	 ax3.xaxis.grid()
	 ax3.yaxis.set_ticks(np.arange(0, 6, 1))

	 sticks = ['', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
	 ax5 = fig.add_subplot(gs[2], sharex=ax3)  #
	 ax5.set_xticks(pos[::sticksDensity] + (width / 2))
	 ax5.set_xticklabels(sticks[:])
	 ax5.set_xlim([0, 24 * 60 / Interval])
	 ax5.set_ylabel('Price ($)')
	 ax5.plot(pos, Prices[:], color='purple')
	 xticklabels = ax.get_xticklabels() + ax2.get_xticklabels()
	 plt.setp(xticklabels, visible=False)
	 plt.subplots_adjust(hspace=0.001)
	 ax5.set_xlabel('Time')
	 ax5.yaxis.grid()
	 ax5.xaxis.grid()
	 ax5.yaxis.set_ticks(np.arange(0, 2, 1))

	 plt.suptitle(str(building)+" | "+str(zone)+" | "+str(date)+" || "+str(method)+"" + ' \n Total Discomfort=' + str(discomfort) + ' $F^2$min')
	 plt.show()
	 return 1


def getData(building, zone, date):

	"""Whatever data we get should be stored.
	date: in PST"""


	root, dirs, files = os.walk("CacheThanos/").next()
	Flag = False
	for index, thefile in enumerate(files, start=1):
		if str(building)+str(zone)+str(date)+".dat" == thefile:
			Flag = True

	if Flag == False:

		# get config
		cfg = utils.get_config(building)
		zone_cfg = utils.get_zone_config(building, zone)

		
		manual = []
		zone_log = utils.get_zone_log(building, zone)
		if zone_log:
			for line in zone_log:
				dateLog = utils.get_mdal_string_to_datetime(line.split(" : ")[1][:-1])
				dateLog = dateLog.astimezone(pytz.timezone("US/Pacific"))
				if dateLog.date() == date.date():
					manual.append( int((dateLog.replace(tzinfo=None) - date.replace(tzinfo=None)).total_seconds()/60) )

		interval = cfg["Interval_Length"]

		# client = utils.choose_client(cfg)
		client = get_client()

		start = date.replace(hour=0, minute=0, second=0)
		end = date.replace(day=date.day + 1, hour=0, minute=0, second=0)

		# Generate utc times. Use UTC for any archiver getting methods.
		pst_pytz = pytz.timezone("US/Pacific")

		start_pst = pst_pytz.localize(start)
		start_utc = start_pst.astimezone(pytz.timezone("UTC"))

		end_pst = pst_pytz.localize(end)
		end_utc = end_pst.astimezone(pytz.timezone("UTC"))

		datamanager = DataManager(cfg, zone_cfg, client, zone, now=start_utc)


		# get setpoints
		ground_truth_setpoints_df = datamanager.thermostat_setpoints(start_utc, end_utc)[zone] # from archiver
		ground_truth_setpoints_df.index = ground_truth_setpoints_df.index.tz_convert(pst_pytz)

		config_setpoints_df = datamanager.better_comfortband(start)
		safety_setpoints_df = datamanager.better_safety(start)

		config_setpoints = config_setpoints_df[["t_low", "t_high"]].values
		safety_setpoints = safety_setpoints_df[["t_low", "t_high"]].values



		# Get tstat and weather data
		thermal_data_manager = ThermalDataManager(cfg, client)

		inside_data, outside_data = utils.get_raw_data(building=building, client=client, cfg=cfg,
			 	 	 	 	 	 	 	   start=start_utc, end=end_utc, force_reload=True)
		zone_inside_data = inside_data[zone]
		zone_inside_data.index = zone_inside_data.index.tz_convert(pst_pytz)
		outside_data = thermal_data_manager._preprocess_outside_data(outside_data.values())
		outside_data.index = outside_data.index.tz_convert(pst_pytz)
		outside_data = outside_data.resample("1T").interpolate()

		Tin = zone_inside_data["t_in"].values
		if np.isnan(Tin).any():
			print "Warning: Tin contains NaN. Estimates are based on interpolations"
			nans, x= nan_helper(Tin)
			Tin[nans]= np.interp(x(nans), x(~nans), Tin[~nans])

		# TODO shitty hack
		# taking the raw data and putting it into a data frame full of nan. Then, interpolating the data to get
		# data for the whole day.
		Tout = pd.DataFrame(columns=["t_out"], index=pd.date_range(start=start, end=end, freq="1T"))
		Tout.index = Tout.index.tz_localize(pst_pytz)
		Tout["t_out"][outside_data.index[0]:outside_data.index[-1]] = outside_data["t_out"]
		Tout = Tout.ffill()["t_out"].values[:1440]

		Policy = zone_inside_data["action"].values


		# Prepare discomfort
		discomfortManager = Discomfort(setpoints=config_setpoints)

		# get occupancies
		occupancy_config = datamanager.better_occupancy_config(start)
		try:
			 occupancy_ground = datamanager.occupancy_archiver(start=start, end=end)
		except:
			 if zone_cfg["Advise"]["Occupancy_Sensors"] == True:
			 	print("Warning, could not get ground truth occupancy.")
			 occupancy_ground = None

		if occupancy_ground is None:
			 occupancy_use = occupancy_config
		else:
			 occupancy_use = occupancy_ground

		occupancy_use = occupancy_use["occ"].values

		discomfort = []
		for i in range(len(Tin)):
			 # for the ith minute
			 assert len(Tin) <= len(occupancy_use)
			 tin = Tin[i]
			 occ = occupancy_use[i]
			 discomfort.append(discomfortManager.disc(t_in=tin, occ=occ, node_time=i, interval=1))


		# get consumption and cost and prices
		prices = datamanager.better_prices(start).values
		heating_consumption = zone_cfg["Advise"]["Heating_Consumption"]
		cooling_consumption = zone_cfg["Advise"]["Cooling_Consumption"]

		energy_manager = EnergyConsumption(prices, interval, now=None,
			 	 	 	 	 	    heat=heating_consumption, cool=cooling_consumption)
		cost = []
		for i in range(len(Policy)):
			 # see it as the ith minute. That's why we need the assert
			 assert len(Policy) <= len(prices)
			 action = Policy[i]
			 cost.append(energy_manager.calc_cost(action=action, time=i))
		cost = np.array(cost)

		# Cache the data and check if already downloaded!
		OPs = occupancy_use[:1440]


		TinsUPComfortBand = config_setpoints_df["t_high"][:1440]

		TinsDOWNComfortBand = config_setpoints_df["t_low"][:1440]

		TinsUPSafety = safety_setpoints_df["t_high"][:1440]

		TinsDOWNSafety = safety_setpoints_df["t_low"][:1440]

		TinsUPsp = ground_truth_setpoints_df["t_high"][:1440]

		TinsDOWNsp = ground_truth_setpoints_df["t_low"][:1440]

		Costs = cost[:1440]

		Prices = prices[:1440]

		Discomforts = discomfort[:1440]

		if zone_cfg["Advise"]["Actuate"]==True:
			if zone_cfg["Advise"]["MPC"]==True:
				method = "MPC ("+str(zone_cfg["Advise"]["Actuate_Start"])+"-"+str(zone_cfg["Advise"]["Actuate_End"])+")"  # get ground truth from config
			else:
				method = "Expansion ("+str(zone_cfg["Advise"]["Actuate_Start"])+"-"+str(zone_cfg["Advise"]["Actuate_End"])+")"  # get ground truth from config
		else:
			method = "We did NOT actuate this zone"

		temp = OPs, Tin, Tout, Policy, TinsUPComfortBand, TinsDOWNComfortBand, TinsUPSafety, TinsDOWNSafety, TinsUPsp, TinsDOWNsp, Costs, Prices, Discomforts, method, manual, building, zone, date
	 	pickle.dump( temp, open( "CacheThanos/"+str(building)+str(zone)+str(Date)+".dat", "wb" ) )
		return temp

	else:
		return pickle.load( open("CacheThanos/"+str(building)+str(zone)+str(date)+".dat", "rb" ) )



if __name__ == "__main__":
	 building, zone = choosebuildingandzone()
	 Date = datetime.datetime(year=2018, month=7, day=10)
	 OPs, Tins, Tout, Policy, TinsUPComfortBand, TinsDOWNComfortBand, TinsUPSafety, TinsDOWNSafety, TinsUPsp, TinsDOWNsp, Costs, Prices, Discomforts, method, manual, building, zone, date = getData(building, zone, Date)
	 PlotDay(OPs, Tins, Tout, Policy, TinsUPComfortBand, TinsDOWNComfortBand, TinsUPSafety, TinsDOWNSafety, TinsUPsp, TinsDOWNsp, Costs, Prices, Discomforts, method, manual, building, zone, date)

	 # import datetime
	 # getData("avenal-recreation-center", zone="HVAC_Zone_Tech_Center", date=datetime.datetime(year=2018, day=10, month=7, minute=12, hour=12))
