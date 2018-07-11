from xbos import get_client
from xbos.services.hod import HodClient
from xbos.devices.thermostat import Thermostat
import datetime
import math
import sys
import threading
import time
import traceback
import pytz
import yaml


#Tstat Brick query (Fixed for missing relationships)
thermostat_query = """SELECT ?zone ?uri FROM  %s WHERE {
          ?tstat rdf:type brick:Thermostat .
          ?tstat bf:controls ?RTU .
          ?RTU rdf:type brick:RTU .
          ?RTU bf:feeds ?zone. 
          ?zone rdf:type brick:HVAC_Zone .
          ?tstat bf:uri ?uri.
          };"""

#Preset of some actions
COOLING_ACTION = {"heating_setpoint": 65, "cooling_setpoint": 68, "override": True, "mode": 3}
HEATING_ACTION = {"heating_setpoint": 80, "cooling_setpoint": 95, "override": True, "mode": 3}
NO_ACTION = {"heating_setpoint": 64, "cooling_setpoint": 75, "override": True, "mode": 3}
PROGRAMMABLE = {"override": False}

#Setter
def writeTstat(tstat, action):
  print("Action we are writing", action)
  print("Tstat uri", tstat._uri)
  tstat.write(action)

#Getter 
def printTstat(tstat):
    try:
        print("heating setpoint", tstat.heating_setpoint)
        print("cooling setpoint", tstat.cooling_setpoint)
        print("temperature", tstat.temperature)
        print("override", tstat.override)
    except:
        print("WARNING: for tstat %s the setpoints could not be read" % tstat)


######################################################################## Main Script:

#Buildings to be affected
# buildings = ["avenal-animal-shelter", "avenal-veterans-hall", "avenal-movie-theatre", "avenal-public-works-yard", "avenal-recreation-center", "orinda-community-center", "north-berkeley-senior-center", "south-berkeley-senior-center"]
# buildings = ["csu-dominguez-hills"]
buildings = ["south-berkeley-senior-center",
             "north-berkeley-senior-center",
             "avenal-veterans-hall",
             "ciee", "orinda-community-center",
             "word-of-faith-cc",
             "jesse-turner-center",
             "orinda-community-center",
             "avenal-recreation-center",
             "avenal-animal-shelter", "avenal-movie-theatre", "avenal-public-works-yard",
             "avenal-recreation-center", "berkeley-corporate-yard"]



# Getting clients
client = get_client()
hc = HodClient("xbos/hod", client)

for BUILDING in buildings:
    print("================================================")
    print("")
    print("Working on building", BUILDING)
    print("")

    query_data = hc.do_query(thermostat_query % BUILDING)["Rows"]
    query_data = [x for x in query_data if x["?zone"]!="HVAC_Zone_Please_Delete_Me"] #TODO CHANGE THE PLEASE DELETE ME ZONE CHECK WHEN FIXED

    try:
        tstats = {d["?zone"]: Thermostat(client, d["?uri"]) for d in query_data}
    except:
        print("Warning: Unable to get Thermostat. Aborting for this building.")
        continue

    ##### RUN
    for zone, tstat in tstats.items():
        writeTstat(tstat, PROGRAMMABLE)

    # wait to let the setpoints get through
    # time.sleep(2)
    # Printing the data for every tstat
    for zone, tstat in tstats.items():
      print("")
      print("Checking zone:", zone)
      print("Checking zone uri:", tstat._uri)
      printTstat(tstat)
      print("Done checking zone", zone)
      print("")

