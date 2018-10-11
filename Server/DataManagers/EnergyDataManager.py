
from __future__ import division

from datetime import timedelta
from xbos import get_client
from xbos.services.pundat import DataClient, make_dataframe
from xbos.services.mdal import *
import numpy as np
import pandas as pd

from xbos.services.hod import HodClient
# from matplotlib.pyplot import step, xlim, ylim, show
import matplotlib.pyplot as plt
import datetime, pytz



now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(tz=pytz.timezone("America/Los_Angeles"))
print now
start = now.strftime('%Y-%m-%d %H:%M:%S %Z')
end = '2017-09-21 00:00:00 PST'
WINDOW= '10min'

# data clients
mdal = MDALClient("xbos/mdal")
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

print("getting lighting")

# resp = mdal.do_query(lighting_meter_query_mdal, timeout=120)
lighting_df = resp['df']

print('got lighting')



heating_consume = .3 # in kW
cooling_consume = 5. # kW
meter = df.columns[0]
all_but_meter = df.columns[1:]

# amount to subtract for heating, cooling
h = (df[all_but_meter] == 1).apply(sum, axis=1) * heating_consume
c = (df[all_but_meter] == 2).apply(sum, axis=1) * cooling_consume

meterdata = df[meter] - h - c

meterdata = meterdata #/(1000)
#print meterdata
print lighting_df.describe()
meterdata = pd.DataFrame.from_records({'House Consumption': meterdata})
print meterdata.describe()
print meterdata['House Consumption']


meterdata = meterdata.tz_convert(pytz.timezone("America/Los_Angeles")) #/??????????????????????????????????

meterdata.to_csv("ConsumptionData")


yesterday = now - datetime.timedelta(hours=12)