from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta, datetime, time
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#rcParams['mathtext.fontset'] = 'custom'
##rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
#rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
import matplotlib.mlab as mlab
import math


df = pd.DataFrame(index=pd.DatetimeIndex(start=datetime(2002,1,1), periods=1440, freq='T'))
df['Europe'] = 0.16
df.loc[(df.index.time > time(hour=13, minute=00)) & (
    df.index.time < time(hour=16, minute=00)), 'Europe'] = 0.107
df.loc[(df.index.time > time(hour=20, minute=00)) & (
    df.index.time < time(hour=22, minute=00)), 'Europe'] = 0.107
df.loc[(df.index.time > time(hour=00, minute=00)) & (
    df.index.time < time(hour=5, minute=00)), 'Europe'] = 0.107
df['US'] = 0.202
df.loc[(df.index.time > time(hour=8, minute=30)) & (
    df.index.time < time(hour=21, minute=30)), 'US'] = 0.230
df.loc[(df.index.time > time(hour=12, minute=00)) & (
    df.index.time < time(hour=18, minute=00)), 'US'] = 0.253
df['US'] = df['US']
dOLLARS=df['US'].values
pOUNDS=df['Europe'].values
print shape(dOLLARS[:])


sticks = []
disc=60*4
for i in range(0, 1441, disc):
	if int(round(i/60))<10:
		hours = "0"+str(int(round(i/60)))
	else:
		hours = str(int(round(i/60)))
	if int(round(i%60))<10:
		minutes = "0"+str(int(round(i%60)))
	else:
		minutes = str(int(round(i%60)))
	sticks.append(hours+":"+minutes)
pos = arange(1441)


##################################################  1
#shift axes
shift=0

fig, ax = plt.subplots()
ind = arange(0,1440)


ax.plot(ind, dOLLARS, '-', label="Import/export tariff, USA")
ax.plot(ind, pOUNDS*1.28, '-', label="Import tariff, UK")
ax.plot(ind, repeat(0.0485,1440)*1.28, '-', label="Export tariff, UK")

ax.set_xticks(pos[::disc])
ax.set_xticklabels(sticks)
ax.set_xlim(0, 1440)
ax.set_ylabel("$/kWh")
ax.legend(loc=4, ncol=3, numpoints=1, fancybox=False, framealpha=0.7)
ax2 = ax.twinx()
ax2.set_ylabel(u'\u00A3'+'/kWh')

#ax.set_ylim(0.0001, 0.35)
#ax2.set_ylim(0.0001/1.28, 0.35/1.28)

plt.show()
'''
##################################################  2
fig, ax = plt.subplots()
ind = arange(0,1440)


#ax.plot(ind, dOLLARS, '-', label="Import/export tariff, USA")
#ax.plot(ind, pOUNDS*1.28, '-', label="Import tariff, UK")
#ax.plot(ind, repeat(0.0485,1440)*1.28, '-', label="Export tariff, UK")

mu = (12+1)*60+38
variance = (0.33*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))

mu = (12+5)*60+40
variance = (0.33*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))


mu = (12+8)*60+38
variance = (0.89*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))

mu = 7*60+40
variance = (0.57*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))


ax.set_xticks(pos[::disc])
ax.set_xticklabels(sticks)
ax.set_xlim(0, 1440)

ax.set_ylabel("$/kWh")
ax.legend(loc=2, ncol=1, numpoints=1, fancybox=False, framealpha=0.7)
ax2 = ax.twinx()
ax2.set_ylabel(u'\u00A3'+'/kWh')

#ax.set_ylim(0.0001, 0.35)
#ax2.set_ylim(0.0001/1.28, 0.35/1.28)

plt.show()
'''

##################################################  1
#shift axes
shift=12
#sticks = ['12:00', '00:00', '12:00', "Plug-in time\n$\mathcal{N}$ ($\mu$=6:38pm, $\sigma$=0.89h)", "Plug-out time\n$\mathcal{N}$ ($\mu$=7:40pm, $\sigma$=0.57h)"]
sticks = ['12:00', '00:00', '12:00', "18:38", "7:40"]
poss = [0,  720, 1440 , (12+6-shift)*60+38, (7+shift)*60+40] #pos[::disc]


fig, ax = plt.subplots()
ind = arange(0,1440)


#ax.plot(ind, dOLLARS, '-', label="Import/export tariff, USA")
#ax.plot(ind, pOUNDS*1.28, '-', label="Import tariff, UK")
#ax.plot(ind, repeat(0.0485,1440)*1.28, '-', label="Export tariff, UK")

mu = (12+6-shift)*60+38
variance = (0.89*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 100*sigma, mu + 100*sigma, 10000)
y=mlab.normpdf(x, mu, sigma)
plt.plot(x,y, color="g")
plt.fill_between(x, 0, y, color='g', alpha=0.4, label="Plug-in time $\mathcal{N}$ ($\mu$=18:38, $\sigma$=0.89h)")


mu = (7+shift)*60+40
variance = (0.57*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 100*sigma, mu + 100*sigma, 10000)
plt.plot(x,y, color="purple")
plt.fill_between(x, 0, y, color='purple', alpha=0.4, label="Plug-out time $\mathcal{N}$ ($\mu$=7:40, $\sigma$=0.57h)")

tilt=0#0.16
plt.arrow((12+6-shift)*60+38, 0.004-0.0006, (((7+shift)*60+40)-((12+6-shift)*60+38)-10), tilt, width=0.0000000001, fc='k', ec='k', head_width=0.0003, head_length=10)

plt.text((12+6-shift)*60+38, 0.004-0.0005, "B$\sim$$\mathcal{U}$(50%, 90%) ", fontsize=12)

plt.text(((7+shift)*60+40), 0.004-0.0005+tilt, r"B$^\theta$=100%", fontsize=12)

#plt.scatter(((7+shift)*60+40), 0.004*100-0.04+0.15, marker='x', s=100, c="r")#'x', c='r')

#ax.xaxis.set_ticks_position('none')
ax.set_yticks(ax.get_yticks()[::2])
ax.set_xticks(poss)
ax.set_xticklabels(sticks)
ax.set_xlim(0, 1440)
ax.set_ylabel("Propability density")
leg = plt.legend(fancybox=True, loc="upper center")
# set the alpha value of the legend: it will be translucent
leg.get_frame().set_alpha(0.7)
#ax2 = ax.twinx()
#ax2.set_ylabel(u'\u00A3'+'/kWh')

#ax.set_ylim(0.0001, 0.35)
#ax2.set_ylim(0.0001/1.28, 0.35/1.28)

#plt.legend()
plt.show()


##################################################  2
#shift axes
shift=12
sticks = ['12:00', '00:00', '12:00', "20:38", "7:40", "13:00", "17:00"]
poss = [0, 720, 1440 , (12+8-shift)*60+38, (7+shift)*60+40, (12+1-shift)*60+38, (12+5-shift)*60+40] #pos[::disc]



fig, ax = plt.subplots()
ind = arange(0,1440)

#ax.plot(ind, dOLLARS, '-', label="Import/export tariff, USA")
#ax.plot(ind, pOUNDS*1.28, '-', label="Import tariff, UK")
#ax.plot(ind, repeat(0.0485,1440)*1.28, '-', label="Export tariff, UK")

mu = (12+1-shift)*60+38
variance = (0.33*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 100*sigma, mu + 100*sigma, 10000)
y=mlab.normpdf(x, mu, sigma)
plt.plot(x,y, color="g")
plt.fill_between(x, 0, y, color='g', alpha=0.4, label="Plug-in time $\mathcal{N}$ ($\mu$=13:00, $\sigma$=0.33)",hatch='\\')

mu = (12+5-shift)*60+40
variance = (0.33*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 100*sigma, mu + 100*sigma, 10000)
y=mlab.normpdf(x, mu, sigma)
plt.plot(x,y, color="purple")
plt.fill_between(x, 0, y, color='purple', alpha=0.4, label="Plug-out time $\mathcal{N}$ ($\mu$=17:00, $\sigma$=0.33h)",hatch='\\')


mu = (12+8-shift)*60+38
variance = (0.89*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 100*sigma, mu + 100*sigma, 10000)
y=mlab.normpdf(x, mu, sigma)
plt.plot(x,y, color="g")
plt.fill_between(x, 0, y, color='g', alpha=0.4, label="Plug-in time $\mathcal{N}$ ($\mu$=20:38, $\sigma$=0.89h)")

mu = (7+shift)*60+40
variance = (0.57*60)**2
sigma = math.sqrt(variance)
x = linspace(mu - 100*sigma, mu + 100*sigma, 10000)
y=mlab.normpdf(x, mu, sigma)
plt.plot(x,y, color="purple")
plt.fill_between(x, 0, y, color='purple', alpha=0.4, label="Plug-out time $\mathcal{N}$ ($\mu$=7:40, $\sigma$=0.57h)")

tilt=0#0.16
plt.arrow((12+8-shift)*60+38, 0.004-0.0006, (((7+shift)*60+40)-((12+8-shift)*60+38)-10), tilt, width=0.0000000001, fc='k', ec='k', head_width=0.0003, head_length=10)
plt.text((12+8-shift)*60+38, 0.004-0.0005, "B$\sim$$\mathcal{U}$(60%, 90%) ", fontsize=12)
plt.text(((7+shift)*60+40), 0.004-0.0005+tilt, r"B$^\theta$=100%", fontsize=12)


plt.arrow((12+1-shift)*60+38, 0.004-0.0006, (((12+5-shift)*60+40)-((12+1-shift)*60+38)-10), tilt, width=0.0000000001, fc='k', ec='k', head_width=0.0003, head_length=10)
plt.text((12+1-shift)*60+38, 0.004-0.0005, "B$\sim$$\mathcal{U}$(80%, 90%) ", fontsize=12)
plt.text(((12+5-shift)*60+40), 0.004-0.0005+tilt, r"B$^\theta$=100%", fontsize=12)


#plt.scatter(((7+shift)*60+40), 0.004*100-0.04+0.15, marker='x', s=100, c="r")#'x', c='r')

#ax.xaxis.set_ticks_position('none')
ax.set_yticks(ax.get_yticks()[::2])
ax.set_xticks(poss)
ax.set_xticklabels(sticks)
ax.set_xlim(0, 1440)
ax.set_ylabel("Propability density")
leg = plt.legend(fancybox=True, loc="upper right", ncol=2)
# set the alpha value of the legend: it will be translucent
leg.get_frame().set_alpha(0.7)
#ax2 = ax.twinx()
#ax2.set_ylabel(u'\u00A3'+'/kWh')

#ax.set_ylim(0.0001, 0.35)
#ax2.set_ylim(0.0001/1.28, 0.35/1.28)

#plt.legend()
plt.show()
