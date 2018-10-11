import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import *
import pickle


########## Init Data
method = "CentralZone_baseline.pckl"
with open (method, 'rb') as fp:
    return_dict = pickle.load(fp)

OPs = [i for i in return_dict["OPs"]]
Tins = [i for i in return_dict["Tins"]]
Policy = [i for i in return_dict["Policy"]]
TinsUP = [i for i in return_dict["TinsUP"]]
TinsDOWN = [i for i in return_dict["TinsDOWN"]]
Costs = [i for i in return_dict["Costs"]]#[sum(return_dict["cost"][:i]) for i in range(1, len(return_dict["cost"])+1)]
Prices = [i for i in return_dict["Prices"]]
Discomforts = [i for i in return_dict["Discomforts"]]#sum(return_dict['discomfort'])
##############################################


def FtoC(x):
	return (x-32)*5/9.

def PlotDay(OPs, Tins, Policy, TinsUP, TinsDOWN, Costs, Prices, Discomforts, method="baseline"):
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
	discomfort = sum(Discomforts)
	Costs = [sum(Costs[:i]) for i in range(1, len(Costs)+1)]
	Tins = [FtoC(i) for i in Tins]
	TinsUP = [FtoC(i) for i in TinsUP]
	TinsDOWN = [FtoC(i) for i in TinsDOWN]


	Interval=1
	sticks = []
	sticksDensity=180/Interval
	for i in range(0, 60*24/Interval, sticksDensity):
		if int(round(i*Interval/60))<10:
			hours = "0"+str(int(round(i*Interval/60)))
		else:
			hours = str(int(round(i*Interval/60)))
		if int(round(i*Interval%60))<10:
			minutes = "0"+str(int(round(i*Interval%60)))
		else:
			minutes = str(int(round(i*Interval%60)))
		sticks.append(hours+":"+minutes)

	pos = arange(60*24/Interval)
	width = 1.0     # gives histogram aspect to the bar diagram

	fig = plt.figure()

	gs = gridspec.GridSpec(3, 1,height_ratios=[4,1,1])#

	ax = fig.add_subplot(gs[0])#
	ax.set_xticks(pos[::sticksDensity] + (width / 2))
	ax.set_xticklabels(sticks)
	ax.set_xlim([0,24*60/Interval])
	ax.plot(pos,Tins[:], label="$T^{ IN}$", color='red')
	ax.plot(pos,TinsUP[:], label="$T^{ UP}$", color='blue')
	ax.plot(pos,TinsDOWN[:], label="$T^{ DOWN}$", color='blue')

	ax4 = ax.twinx()
	ax4.plot(pos,Costs[:], color='green')
	ax4.set_ylabel('Cost ($)')

	ax.set_ylim(0.1, 2500000)
	ax.set_ylabel(r"Temperature ($^\circ$C)")
	ax.set_ylim(0.1, 35)
	ax.xaxis.grid()
	ax.yaxis.grid()


	ax2 = ax.twinx()
	ax2.set_xticks(pos[::sticksDensity] + (width / 2))
	ax2.set_xticklabels(sticks)
	ax2.set_xlim([0,24*60/Interval])
	ax2.plot(0,0, label="$T^{ IN}$", color='red')
	ax2.plot(0,0, label="Cost", color='green')
	ax2.plot(0,0, label="Comfort-band limits", color='blue')
	ax2.plot(0,0, label="HVAC state", color='orange')
	ax2.plot(0,0, label="Prices", color='purple')

	ax2.bar(pos, OPs, width, color='grey', alpha=0.4, label="Occupancy", linewidth=0)
	#ax2.bar(0, 0, 0, color='grey', alpha=0.7, label="Occupancy", linewidth=0)
	ax2.legend(loc=2, ncol=6)
	ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=6)#, fancybox=True, shadow=True)
	#ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol64, fancybox=True, shadow=True)
	#ax2.set_ylim(2, 27)
	group_labels1 = ['']
	ax2.set_yticklabels(group_labels1)
	ax2.yaxis.set_ticks(arange(0, 1, 1))



	ax3 = fig.add_subplot(gs[1], sharex=ax)#
	#ax3.set_xticks(pos[::sticksDensity] + (width / 2))
	#ax3.set_xticklabels(sticks)
	ax3.set_xlim([0,24*60/Interval])
	ax3.set_ylabel('Action')
	ax3.plot(pos,Policy[:], label="Policy of Function", color='orange')
	#xticklabels = ax.get_xticklabels()+ax2.get_xticklabels()
	#plt.setp(xticklabels, visible=False)
	plt.subplots_adjust(hspace=0.001)
	#ax3.set_xlabel('Time')
	ax3.set_ylim(-1, 4)
	group_labels = ['Nothing', 'Heating', 'Cooling', 'Ventilation']
	ax3.set_yticklabels(group_labels)
	ax3.yaxis.grid()
	ax3.xaxis.grid()
	ax3.yaxis.set_ticks(arange(0, 4, 1))

	sticks = ['', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
	ax5 = fig.add_subplot(gs[2], sharex=ax3)#
	ax5.set_xticks(pos[::sticksDensity] + (width / 2))
	ax5.set_xticklabels(sticks[:])
	ax5.set_xlim([0,24*60/Interval])
	ax5.set_ylabel('Price ($)')
	ax5.plot(pos,Prices[:], color='purple')
	xticklabels = ax.get_xticklabels()+ax2.get_xticklabels()
	plt.setp(xticklabels, visible=False)
	plt.subplots_adjust(hspace=0.001)
	ax5.set_xlabel('Time')
	ax5.yaxis.grid()
	ax5.xaxis.grid()
	ax5.yaxis.set_ticks(arange(0, 2, 1))

	plt.suptitle(method+' - Total Discomfort='+str(discomfort)+' $F^2$min')
	plt.show()
	return 1


PlotDay(OPs, Tins, Policy, TinsUP, TinsDOWN, Costs, Prices, Discomforts, method)
