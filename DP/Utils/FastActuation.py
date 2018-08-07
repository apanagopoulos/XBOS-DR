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

sys.path.append("../Server")
import utils


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
HEATING_ACTION = {"heating_setpoint": 70, "cooling_setpoint": 75, "override": True, "mode": 3}
NO_ACTION = {"heating_setpoint": 66, "cooling_setpoint": 73, "override": True, "mode": 3}
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
        print("action", tstat.state)
        print("override", tstat.override)
    except:
        print("WARNING: for tstat %s the setpoints could not be read" % tstat)


######################################################################## Main Script:

#Buildings to be affected
# buildings = ["avenal-animal-shelter", "avenal-veterans-hall", "avenal-movie-theatre", "avenal-public-works-yard", "avenal-recreation-center", "orinda-community-center", "north-berkeley-senior-center", "south-berkeley-senior-center"]
# buildings = ["csu-dominguez-hills"]
# buildings = ["south-berkeley-senior-center",
#              "north-berkeley-senior-center",
#              "avenal-veterans-hall",
#              "ciee", "orinda-community-center",
#              "word-of-faith-cc",
#              "jesse-turner-center",
#              "orinda-community-center",
#              "avenal-recreation-center",
#              "avenal-animal-shelter", "avenal-movie-theatre", "avenal-public-works-yard",
#              "avenal-recreation-center", "berkeley-corporate-yard"]

buildings = ["jesse-turner-center"]

BUILDING = "jesse-turner-center"

cfg_building = utils.get_config(BUILDING)


# if end < now:
#     wait_seconds = 0
# else:
#     wait_seconds = (end - now).seconds
# print("Waiting for %f seconds." % wait_seconds)
# time.sleep(wait_seconds)

# Getting clients
client = utils.choose_client(cfg_building)
hc = HodClient("xbos/hod", client)

print("================================================")
print("")
print("Working on building", BUILDING)
print("")

query_data = hc.do_query(thermostat_query % BUILDING)["Rows"]
query_data = [x for x in query_data if
              x["?zone"] != "HVAC_Zone_Please_Delete_Me"]  # TODO CHANGE THE PLEASE DELETE ME ZONE CHECK WHEN FIXED

try:
    tstats = {d["?zone"]: Thermostat(client, d["?uri"]) for d in query_data}
except:
    raise Exception("Warning: Unable to get Thermostat. Aborting this building.")


berkeley_timezone = pytz.timezone("America/Los_Angeles")

now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(berkeley_timezone)

start = now.replace(hour=14, minute=0, second=0, microsecond=0)
end = now.replace(hour=18, minute=0, second=0, microsecond=0)

# start = now.replace(hour=13, minute=47, second=0, microsecond=0)
# end = now.replace(hour=13, minute=55, second=0, microsecond=0)

actuated_once = False

run_program = True
while run_program:
    iteration_start = time.time()


    # set wether to actuate
    actuate = (start <= now <= end) and (not actuated_once)

    print("=============================================")
    print("Acutation: %f with now to start: %f" % (int(actuate), (start - now).seconds))

    if actuate:
        ##### RUN
        for zone, tstat in tstats.items():
            if "Basketball" in zone:
                heating_setpoint = tstat.heating_setpoint
                cooling_setpoint = tstat.cooling_setpoint

                new_cooling_setpoint = 75

                # checking if heating setpoint is reasonable
                if heating_setpoint > new_cooling_setpoint:
                    new_heating_setpoint = 70
                else:
                    new_heating_setpoint = heating_setpoint

                action_to_write = {"heating_setpoint": new_heating_setpoint, "cooling_setpoint": new_cooling_setpoint,
                                    "override": True, "mode": 3}
                print("We are writing the following action: ", action_to_write)
                writeTstat(tstat, action_to_write)

        actuated_once = True

        # wait to let the setpoints get through
        time.sleep(5)

    # Printing the data for every tstat
    for zone, tstat in tstats.items():
      print("")
      print("Checking zone:", zone)
      print("Checking zone uri:", tstat._uri)
      printTstat(tstat)
      print("Done checking zone", zone)
      print("")


    WAIT_MINUTES = 15

    # wait for next iteration
    wait_time = WAIT_MINUTES * 60 - (time.time() - iteration_start)

    # set the appropriate wait time if it would be less to get to start or end
    if wait_time > (start - now).seconds and start > now:
        wait_time = (start - now).seconds
    if wait_time > (end - now).seconds and end > now:
        wait_time = (end - now).seconds
    if not actuate and start < now:
        wait_time = 0
    if actuate and end < now:
        wait_time = 0

    # just to make sure we are waiting a bit
    wait_time = max(1, wait_time)

    print("wait time", wait_time)
    time.sleep(wait_time)

    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(berkeley_timezone)

    run_program = now < end

print("ENDING PROGRAM")

for zone, tstat in tstats.items():
    if "Basketball" in zone:
        writeTstat(tstat, PROGRAMMABLE)

time.sleep(5)
# Printing the data for every tstat
for zone, tstat in tstats.items():
    print("")
    print("Checking zone:", zone)
    print("Checking zone uri:", tstat._uri)
    printTstat(tstat)
    print("Done checking zone", zone)
    print("")

