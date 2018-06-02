# coding: utf-8

# In[3]:


import datetime
import time
from collections import defaultdict
import pickle
import os

import numpy as np
import pandas as pd
from xbos import get_client
from xbos.devices.thermostat import Thermostat
from xbos.services.hod import HodClient


# In[4]:

class ThermalDataCollector:
    def __init__(self, client, building, safemode):

        thermostat_query = """SELECT ?zone ?uri FROM %s WHERE { 
                  ?tstat rdf:type brick:Thermostat .
                  ?tstat bf:hasLocation/bf:isPartOf ?location_zone .
                  ?location_zone rdf:type brick:HVAC_Zone .
                  ?tstat bf:controls ?RTU .
                  ?RTU rdf:type brick:RTU . 
                  ?RTU bf:feeds ?zone. 
                  ?zone rdf:type brick:HVAC_Zone . 
                  ?tstat bf:uri ?uri.
                };"""

        # Start of FIX for missing Brick query
        self.thermostat_query = """SELECT ?zone ?uri FROM  %s WHERE {
                  ?tstat rdf:type brick:Thermostat .
                  ?tstat bf:controls ?RTU .
                  ?RTU rdf:type brick:RTU .
                  ?RTU bf:feeds ?zone. 
                  ?zone rdf:type brick:HVAC_Zone .
                  ?tstat bf:uri ?uri.
                  };"""
        # End of FIX - delete when Brick is fixed

        self.client = client

        self.COOLING_ACTION = lambda tstat: {"heating_setpoint": 50, "cooling_setpoint": 65, "override": True,
                                             "mode": 3}
        self.HEATING_ACTION = lambda tstat: {"heating_setpoint": 80, "cooling_setpoint": 95, "override": True,
                                             "mode": 3}
        self.NO_ACTION = lambda tstat: {"heating_setpoint": tstat.temperature - 5,
                                        "cooling_setpoint": tstat.temperature + 5,
                                        "override": True, "mode": 3}
        self.building = building
        self.safemode = safemode

        self.tstats = self.getTstats()


    def gatherZoneData(self, tstat):
        data = {"heating_setpoint": tstat.heating_setpoint,
                "cooling_setpoint": tstat.cooling_setpoint,
                "state": tstat.state,
                "temperature": tstat.temperature}
        return data

    def getTstats(self):
        hod_client = HodClient("xbos/hod", self.client)
        thermostat_query_data = hod_client.do_query(self.thermostat_query % self.building)["Rows"]
        return {tstat["?zone"]: Thermostat(client, tstat["?uri"]) for tstat in thermostat_query_data}

    def controlZone(self, tstats, action_messages, interval, dt, flag_function=lambda: True,
                   stop_time=datetime.datetime.utcnow() + datetime.timedelta(days=1)):
        """
        :param tstats: {zone: tstat object}
        :param action_messages: {zone: action dictionary}
        :param interval: how long to execute action in minutes
        :param dt: how often to record data and rewrite message in minutes
        :param flag_function: a function with no parameters that can also halt the while loop besides reaching the interval length. 
                        For now we use functions which determine whether we have reached setpoint temperatures.
        :param stop_time: Time at which to stop everything. In UTC. Usually when work begins. 
        returns: {zone: pd.df columns:["heating_setpoint",
                  "cooling_setpoint",
                  "state",
                  "temperature", "dt"] index=time right after all actions were written to thermostats (freq=dt)} AND
                  (array) (expected cooling_setpoint, recorded cooling_setpoint,
                   expected heating_setpoint, recorded heating set_point, time, zone) records if someone was interfering with the setpoints."""
        start_time = time.time()
        recorded_data = defaultdict(list)
        recorded_setpoint_changes = []
        while time.time() - start_time < 60 * interval and flag_function() and (datetime.datetime.utcnow() < stop_time):
            try:
                # potential improvement is to make the times more accurate
                run_time = time.time()
                for zone, action in action_messages.items():
                    action_msg = action(tstats[zone])
                    if self.safemode is False:
                        tstats[zone].write(action_msg)

                    # recording if the setpoint we are changing the thermostat to is different than the existing
                    # setpoint. Important to see if someone is changing the setpoints manually.
                    tstat = tstats[zone]
                    if tstat.heating_setpoint != action_msg["heating_setpoint"] or tstat.cooling_setpoint != action_msg[
                        "cooling_setpoint"]:
                        recorded_setpoint_changes.append((action_msg["cooling_setpoint"], tstat.cooling_setpoint,
                                                          action_msg["heating_setpoint"], tstat.heating_setpoint,
                                                          datetime.datetime.utcnow(), zone))
                        # TODO uncomment when wanting to check for manual changes.
                        # if clause to not account for the first time we are changing the temperature.
                        # if start_time + 60*dt < time.time():
                        #     print("==============================")
                        #     print("WARNING, manual setpoint changes are happening. Reverting to normal schedule.")
                        #     tstat.write({"override": False})
                        #     return None, recorded_setpoint_changes

                #  using dt as we assume it will be, (i.e. runtime less than dt). We can infer later if it differs.
                time_data = {"time": datetime.datetime.utcnow(), "dt": dt}
                for zone, tstat in tstats.items():
                    row = self.gatherZoneData(tstat)
                    row.update(time_data)
                    recorded_data[zone].append(row)
            except:
                import traceback
                print(traceback.format_exc())
                print("Error writing to Thermostat at time", datetime.datetime.utcnow())

            # usually iteration of loop takes less than 0.1 seconds.
            if dt * 60 - (time.time() - run_time) < 0:
                print("Warning: An iteration of the loop took too long. At utc_time: ", time_data["time"])
            time.sleep(max(dt * 60 - (time.time() - run_time), 0))

        dataframe_data = {}
        for zone, data in recorded_data.items():
            data = pd.DataFrame(data).set_index('time')
            dataframe_data[zone] = data
        return dataframe_data, recorded_setpoint_changes

    def main(self, interval_function=lambda action: 30, dt=1):
        """stores the data and calls the controlZone function to control the zones.
        Stores data for each action as {"zone": zone, "action": "action", "data":data} since we are only affecting one zone
        at a time for now. 
        :param interval_function: a function which takes an action as parameter to determine the time for which the action 
                should run in the zone.
        :param dt: the delta time between thermostat writes/reads."""
        zone_order = self.tstats.keys()  # establishes order in which to perform actions.

        actions = {"0": self.NO_ACTION, "1": self.HEATING_ACTION,
                        "2": self.COOLING_ACTION}  # in dictionary so we can shuffle easier if wanted.

        action_order = ["0", "2", "1"]

        # control one zone. All others do nothing.
        for action_zone in zone_order:
            for i in range(3):

                self.tstats = self.getTstats() # making sure we have fresh tstat objects.

                print("Started action %s in zone %s at time %s" % (
                str(action_order[i]), action_zone, datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')))
                # re setting since I want to store data all the time. Just to make sure we aren't loosing anything.

                action = actions[action_order[i]]

                # set action for each zone
                action_messages = {}
                for zone in zone_order:
                    if zone == action_zone:
                        action_messages[zone] = action
                    else:
                        action_messages[zone] = actions[action_order[i]]  # no action
                zone_tstat = self.tstats[action_zone]

                interval = interval_function(i)
                zone_action_msg = action_messages[action_zone]
                # calling the controlZone loop. The function will take over and write and read to and from the zone.
                # TODO Get rid of the ugly lambda function. right now it looks for the tstat and action which we defined in this
                # TODO envrionment and queries it everytime the lambda is called. Not nice code.
                action_data, recorded_setpoint_changes = self.controlZone(self.tstats, action_messages, interval, dt,
                                                                         lambda: (not ((zone_action_msg(zone_tstat)[
                                                                                            "heating_setpoint"] + 2) < zone_tstat.temperature < (
                                                                                       zone_action_msg(zone_tstat)[
                                                                                           "cooling_setpoint"] - 2))) or (
                                                                                 i == 0))
                if action_data is None:
                    print("The recorded setpoint changes: ", recorded_setpoint_changes)
                    print("Stopping ThermalDataCollection.")
                    return


                # sets the action field in the data that was returned. So we know which action was executed.
                for zone, df in action_data.items():
                    if zone == action_zone:
                        df["action"] = np.ones(df.shape[0]) * i
                    else:
                        df["action"] = np.ones(df.shape[0]) * 0

                # prints the differeing setpoints, if there are any.
                if recorded_setpoint_changes != []:
                    print("The following were recorded setpoint changes:", recorded_setpoint_changes)


                # storing the data
                # Create foler if it does not exist yet.
                if not os.path.exists("./Freezing_"+self.building):
                    os.makedirs("./Freezing_"+self.building)
                    print("created directory named: ", "./Freezing_"+self.building)

                with open("./Freezing_"+self.building+ "/"+ str(i) + ";"+  action_zone + ";" + datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'), "wb") as f:
                    pickle.dump({"zone": action_zone, "action": i, "data": action_data}, f)

                print("Done with action: ", i)

            print("done with zone", action_zone)


if __name__ == '__main__':
    import sys

    Server = True
    Safemode = False
    try:
        Building = sys.argv[1]
    except:
        sys.exit("Please specify the building name as an argument")

    if Server == True:
        Entity_File = "../thanos.ent"
        Agent_IP = '172.17.0.1:28589'
        client = get_client(Agent_IP, Entity_File)
    else:
        client = get_client()

    '''
    avenal-veterans-hall
    avenal-public-works-yard
    avenal-animal-shelter
    avenal-movie-theatre
    avenal-recreation-center
    '''
    # Building = "avenal-veterans-hall"

    collector = ThermalDataCollector(client, Building, Safemode)

    interval_function = lambda action: 90 if action != 0 else 15
    collector.main(interval_function, dt=15)
