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

# Tstat Brick query (Fixed for missing relationships)
thermostat_query = """SELECT ?zone ?uri FROM  %s WHERE {
          ?tstat rdf:type brick:Thermostat .
          ?tstat bf:controls ?RTU .
          ?RTU rdf:type brick:RTU .
          ?RTU bf:feeds ?zone. 
          ?zone rdf:type brick:HVAC_Zone .
          ?tstat bf:uri ?uri.
          };"""

room_type_query = """
    SELECT ?room ?label ?zone FROM %s WHERE {
        ?room rdf:type brick:Room.
	    ?room rdf:label ?label.
  		?room bf:isPartOf ?zone.
  		?zone rdf:type brick:HVAC_Zone
	};"""

# Preset of some actions
COOLING_ACTION = {"heating_setpoint": 65, "cooling_setpoint": 68, "override": True, "mode": 3}
HEATING_ACTION = {"heating_setpoint": 70, "cooling_setpoint": 75, "override": True, "mode": 3}
NO_ACTION = {"heating_setpoint": 66, "cooling_setpoint": 73, "override": True, "mode": 3}
PROGRAMMABLE = {"override": False}


# Setter
def writeTstat(tstat, action):
    print("Action we are writing", action)
    print("Tstat uri", tstat._uri)
    tstat.write(action)


# Getter
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

# Buildings to be affected
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

BUILDING = "csu-dominguez-hills"

cfg_building = utils.get_config(BUILDING)


# if end < now:
#     wait_seconds = 0
# else:
#     wait_seconds = (end - now).seconds
# print("Waiting for %f seconds." % wait_seconds)
# time.sleep(wait_seconds)

debug = False

# Getting clients
client = get_client()
# if debug:
#     client = utils.choose_client()
# else:
#     client = utils.choose_client(cfg_building)

hc = HodClient("xbos/hod", client)

print("================================================")
print("")
print("Working on building", BUILDING)
print("")

query_data = hc.do_query(thermostat_query % BUILDING)["Rows"]
query_data = [x for x in query_data if
              x["?zone"] != "HVAC_Zone_Please_Delete_Me"]  # TODO CHANGE THE PLEASE DELETE ME ZONE CHECK WHEN FIXED

room_types = hc.do_query(room_type_query % BUILDING)["Rows"]
zone_contain_classroom = {}
for row in room_types:
    curr_zone = row["?zone"]
    # hardcoding a fix. this is how we get zones from tstats....
    curr_zone = curr_zone[:13] + "_" + curr_zone[14:]
    curr_zone = curr_zone.upper()

    if curr_zone not in zone_contain_classroom:
        zone_contain_classroom[curr_zone] = False
    if "\"" in row["?label"]:
        zone_contain_classroom[curr_zone] = True

print(zone_contain_classroom)

try:
    temp_tstats = {d["?zone"]: Thermostat(client, d["?uri"]) for d in query_data}
except:
    raise Exception("Warning: Unable to get Thermostat. Aborting this building.")

tstats = {}
for iter_zone, iter_tstat in temp_tstats.items():
    curr_zone = iter_zone
    curr_zone = curr_zone[:13] + "_" + curr_zone[14:]
    curr_zone = curr_zone.upper()
    tstats[curr_zone] = iter_tstat

print(tstats)

berkeley_timezone = pytz.timezone("America/Los_Angeles")
now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(berkeley_timezone)
start = now.replace(hour=14, minute=0, second=0, microsecond=0)
end = now.replace(hour=18, minute=0, second=0, microsecond=0)

last_cooling_action_written = {}

zone_failed = {}


run_program = True
while run_program:
    iteration_start = time.time()

    if debug:
        print("Last cooling actions", last_cooling_action_written)

    # set wether to actuate
    actuate = start <= now <= end

    print("=============================================")
    print("Acutation: %f with now: %s" % (int(actuate), utils.get_datetime_to_string(now)))

    if actuate:
        ##### RUN
        for zone, tstat in tstats.items():
            try:
                # if not zone_contain_classroom[zone]:
                if True:
                    print("\n ------------- ")
                    print("Writing for zone %s" % zone)
                    heating_setpoint = tstat.heating_setpoint
                    cooling_setpoint = tstat.cooling_setpoint

                    # basically want to see if we can modify the setpoints.
                    if zone in last_cooling_action_written:
                        last_written_cooling = last_cooling_action_written[zone]
                    else:
                        last_cooling_action_written[zone] = None
                        last_written_cooling = None

                    if last_written_cooling != cooling_setpoint:
                        last_cooling_action_written[zone] = None
                        last_written_cooling = None

                    # only if cooling setpoint is less than 80 we do something
                    if cooling_setpoint < 80 and last_written_cooling is None:
                        new_cooling_setpoint = 4 + cooling_setpoint

                        action_to_write = {"heating_setpoint": heating_setpoint, "cooling_setpoint": new_cooling_setpoint,
                                           "override": True, "mode": 3}
                        print("We are writing the following action: ", action_to_write)
                        if not debug:
                            writeTstat(tstat, action_to_write)

                        last_cooling_action_written[zone] = new_cooling_setpoint
                    else:
                        # if last_written_cooling is None:
                        #     float_last_written_cooling = -1
                        # else:
                        #     float_last_written_cooling = last_written_cooling
                        print("No action to write for this zone because cooling setpoint is %f while the last written setpoint is %s" % (cooling_setpoint, str(last_written_cooling)))

                zone_failed[zone] = False
            except:
                print("Zone %s had an exception in writing cooling/heating." % zone)
                zone_failed[zone] = True

        # wait to let the setpoints get through
        time.sleep(30)

        print("\n ++++++ Setting override to false so we can get the schedule from the buildings later. ++++++")
        # wait for a couple of seconds to let the setpoints get set and then set override to false
        for zone, tstat in tstats.items():
            try:
                # if not zone_contain_classroom[zone]:
                if True:
                    if not debug:
                        writeTstat(tstat, PROGRAMMABLE)
                zone_failed[zone] = False
            except:
                print("Zone %s had an exception in writing override." % zone)
                zone_failed[zone] = True




    # Printing the data for every tstat
    for zone, tstat in tstats.items():
        try:
            print("")
            print("Checking zone:", zone)
            print("Checking zone uri:", tstat._uri)
            printTstat(tstat)
            print("Done checking zone", zone)
            print("")
            zone_failed[zone] = False
        except:
            print("Zone %s had an exception in reading." % zone)
            zone_failed[zone] = True


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

for zone, tstat in tstats.items():
    # if not zone_contain_classroom[zone]:
    if True:
        try:
            if not debug:
                writeTstat(tstat, PROGRAMMABLE)
            zone_failed[zone] = False

        except:
            print("Zone %s had an exception in writing override." % zone)
            zone_failed[zone] = True
