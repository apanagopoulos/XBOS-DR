from __future__ import division

from datetime import timedelta
from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe
from house import IEC
from xbos.services.mdal import *
import numpy as np
import pandas as pd

from xbos.services.hod import HodClient
# from matplotlib.pyplot import step, xlim, ylim, show
import matplotlib.pyplot as plt
import datetime, pytz

















'''
from xbos import get_client
from xbos.services.mdal import *
from xbos.services.hod import HodClient
import pandas as pd
import pytz
from sklearn.metrics import mean_squared_error
from dateutil import rrule
from datetime import datetime, timedelta

# data clients
mdal = BOSSWAVEMDALClient("xbos/mdal")
hod = HodClient("xbos/hod")

# temporal parameters
SITE = "ciee"

# Brick queries
building_meters_query = """SELECT ?meter ?meter_uuid FROM %s WHERE {
    ?meter rdf:type brick:Building_Electric_Meter .
    ?meter bf:uuid ?meter_uuid .
};"""
thermostat_state_query = """SELECT ?tstat ?status_uuid FROM %s WHERE {
    ?tstat rdf:type brick:Thermostat_Status .
    ?tstat bf:uuid ?status_uuid .
};"""
lighting_state_query = """SELECT ?lighting ?state_uuid FROM %s WHERE {
    ?light rdf:type brick:Lighting_State .
    ?light bf:uuid ?state_uuid
};"""
lighting_meter_query = """SELECT ?lighting ?meter_uuid FROM %s WHERE {
    ?meter rdf:type brick:Electric_Meter .
    ?lighting rdf:type brick:Lighting_System .
    ?lighting bf:hasPoint ?meter .
    ?meter bf:uuid ?meter_uuid
};"""


targetday="2018-02-01 00:00:00 PST" 
WINDOW="30m" 
N_DAYS=10



T0 = "2017-07-01 00:00:00 PST"
day = datetime.strptime(targetday, "%Y-%m-%d %H:%M:%S %Z")
day = pytz.timezone('US/Pacific').localize(day)
T1 = (day - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S %Z")
tomorrow = (day + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S %Z")

today_start = targetday
today_end = (day + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S %Z")

# retrieve data
building_meters_query_mdal = {
"Composition": ["meter","tstat_state"],
"Selectors": [MEAN, MAX],
"Variables": [
    {
	"Name": "meter",
	"Definition": building_meters_query % SITE,
	"Units": "kW"
    },
    {
	"Name": "tstat_state",
	"Definition": thermostat_state_query % SITE,
    }
    ],
"Time": {
	"T0": T0, "T1": T1,
	"WindowSize": WINDOW,
	"Aligned": True,
    }
}
resp = mdal.do_query(building_meters_query_mdal)
df = resp['df']

consumption_today_sofar = {
"Composition": ["meter"], "Selectors": [MEAN],
"Variables": [{
	"Name": "meter",
	"Definition": building_meters_query % SITE,
	"Units": "kW"
    }],
"Time": {
    "T0": today_start,
    "T1": today_end,
    "WindowSize": WINDOW,
    "Aligned": True,
}
}
resp = mdal.do_query(consumption_today_sofar)
sample = resp['df']

lighting_meter_query_mdal = {
"Composition": ["lighting"],
"Selectors": [MEAN],
"Variables": [
    {
	"Name": "lighting",
	"Definition": lighting_meter_query % SITE,
	"Units": "kW"
    },
    ],
"Time": {
	"T0": T0, "T1": T1,
	"WindowSize": WINDOW,
	"Aligned": True,
    }
}
resp = mdal.do_query(lighting_meter_query_mdal, timeout=120)
lighting_df = resp['df']

# The first column of our DataFrame contains the average building meter data.  We want to 
# subtract from that column the energy consumed when thermostats are in heating or cooling mode. 
# If the thermostat mode column is 1, then the thermostat is heating. If it is 2, then the 
# thermostat is cooling. We are fudging how to handle the 'statistical summary' of a thermostat 
# state by using max(); more sophisticated methods may do a linear scale based on the mean value.

# We use the following values for power consumed: heating (.3 kW) and cooling (5 kW)

heating_consume = .3 # in kW
cooling_consume = 5. # kW
meter = df.columns[0]
all_but_meter = df.columns[1:]

# amount to subtract for heating, cooling
h = (df[all_but_meter] == 1).apply(sum, axis=1) * heating_consume
c = (df[all_but_meter] == 2).apply(sum, axis=1) * cooling_consume #[TODO] change this to count

meterdata = df[meter]  - h - c

'''







######################################################################################################

now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(tz=pytz.timezone("America/Los_Angeles"))
print now
start = now.strftime('%Y-%m-%d %H:%M:%S %Z')
end = '2017-09-21 00:00:00 PST'
WINDOW= '1min'

# data clients
mdal = BOSSWAVEMDALClient("xbos/mdal")
hod = HodClient("xbos/hod")

# temporal parameters
SITE = "ciee"

# Brick queries
building_meters_query = """SELECT ?meter ?meter_uuid FROM %s WHERE {
    ?meter rdf:type brick:Building_Electric_Meter .
    ?meter bf:uuid ?meter_uuid .
};"""
thermostat_state_query = """SELECT ?tstat ?status_uuid FROM %s WHERE {
    ?tstat rdf:type brick:Thermostat_Status .
    ?tstat bf:uuid ?status_uuid .
};"""
lighting_state_query = """SELECT ?lighting ?state_uuid FROM %s WHERE {
    ?light rdf:type brick:Lighting_State .
    ?light bf:uuid ?state_uuid
};"""
lighting_meter_query = """SELECT ?lighting ?meter_uuid FROM %s WHERE {
    ?meter rdf:type brick:Electric_Meter .
    ?lighting rdf:type brick:Lighting_System .
    ?lighting bf:hasPoint ?meter .
    ?meter bf:uuid ?meter_uuid
};"""


building_meters_query_mdal = {
"Composition": ["meter","tstat_state"],
"Selectors": [MEAN, MAX],
"Variables": [
    {
	"Name": "meter",
	"Definition": building_meters_query % SITE,
	"Units": "kW"
    },
    {
	"Name": "tstat_state",
	"Definition": thermostat_state_query % SITE,
    }
    ],
"Time": {
	"T0": start, "T1": end,
	"WindowSize": WINDOW,
	"Aligned": True,
    }
}
resp = mdal.do_query(building_meters_query_mdal)
df = resp['df']


demand= "4d6e251a-48e1-3bc0-907d-7d5440c34bb9"




'''
c = get_client()
archiver = DataClient(c)



uuids = [demand]



print "YOOOO"
'''

lighting_meter_query_mdal = {
	"Composition": ["lighting"],
	"Selectors": [MEAN],
	"Variables": [
	    {
		"Name": "lighting",
		"Definition": lighting_meter_query % SITE,
		"Units": "kW"
	    },
	    ],
	"Time": {
		"T0": start, "T1": end,
		"WindowSize": WINDOW,
		"Aligned": True,
	    }
    }
resp = mdal.do_query(lighting_meter_query_mdal, timeout=120)
lighting_df = resp['df']




heating_consume = .3 # in kW
cooling_consume = 5. # kW
meter = df.columns[0]
all_but_meter = df.columns[1:]

# amount to subtract for heating, cooling
h = (df[all_but_meter] == 1).apply(sum, axis=1) * heating_consume
c = (df[all_but_meter] == 2).apply(sum, axis=1) * cooling_consume

meterdata = df[meter]  - h - c

meterdata = meterdata/(1000*60)
#print meterdata
print lighting_df.describe()
meterdata = pd.DataFrame.from_records({'House Consumption': meterdata})
print meterdata.describe()
print meterdata['House Consumption']



meterdata = meterdata.tz_convert(pytz.timezone("America/Los_Angeles")) #/??????????????????????????????????
yesterday = now - datetime.timedelta(hours=12)

prediction = IEC(meterdata[:yesterday], prediction_window=12*60).predict(["Baseline Finder"])
#prices.index = data.index
#prices['US'] = Dollar/Kwh

index = np.arange(12*60)
plt.plot(index,prediction[["Baseline Finder"]], label="Energy Prediction")
plt.plot(index,meterdata[["House Consumption"]][-12*60:], label="Ground Truth")
plt.xlabel('Predictive horizon (Minutes)')
plt.ylabel(r'KWh')
plt.legend()
plt.show()
#data * prices = posa plirwses
