from collections import defaultdict
from datetime import timedelta
import datetime
import pytz

import pandas as pd
from xbos.devices.light import Light
from xbos.services import mdal
from xbos.services.hod import HodClient


############################################### Initializing our datasets

def lights(building, client, actuate=False):
    ACTUATE = actuate
    SITE = building  # Cahnge this according to the site

    # print RoomsType.RoomsType[df['column_name'] == some_value]
    c = client
    hod = HodClient("xbos/hod")

    skipped = ""

    ####### occ query

    occ_query = """SELECT * FROM %s WHERE {
	?l rdf:type brick:Lighting_System .
	?l bf:feeds ?room .
	?room rdf:type brick:Room .
	?l bf:uri ?luri .
	?l bf:hasPoint ?p .
	?p rdf:type brick:Occupancy_Sensor .
	?p bf:uri ?puri .
	?p bf:uuid ?puuid
	};"""

    res2 = hod.do_query(occ_query % SITE)

    byrooms_o = defaultdict(list)
    for item in res2['Rows']:
        byrooms_o[item['?room']].append(item['?puuid'])

    print "Occupancy uuids loaded!"

    ####### light query


    light_query = """
	SELECT ?light ?room ?light_uri FROM %s WHERE {
	  ?light rdf:type brick:Lighting_System .
	  ?light bf:feeds ?room .
	  ?room rdf:type brick:Room .
	  ?light bf:uri ?light_uri
	};
	"""

    res = hod.do_query(light_query % SITE)

    byrooms = defaultdict(list)
    all_rooms = set()
    for item in res['Rows']:
        all_rooms.add(item['?room'])
        try:
            l = Light(c, item['?light_uri'])
            byrooms[item['?room']].append(l)
        except:
            skipped += str(item['?light']) + "in room" + str(item['?room']) + " skipped! \n"
            pass

    print "Lighting systems loaded!"

    ####### room type query


    type_query = """SELECT * FROM %s WHERE {
	    ?room rdf:type brick:Room.
	    ?room rdf:label ?label
	};
	"""

    res3 = hod.do_query(type_query % SITE)

    byrooms_type = defaultdict(list)
    for item in res3['Rows']:
        byrooms_type[item['?room']].append(item['?label'])

    print byrooms_type
    print "Room types loaded!"

    Actuated = ""
    ################################################ Controls

    c = mdal.MDALClient("xbos/mdal")

    for room in all_rooms:
        Type = byrooms_type[room][0]

        if Type == "Hallway":
            for light in byrooms[room]:
                if ACTUATE:
                    brightness = 20
                    Actuated += "Lights in room" + str(room) + " was set to" + str(brightness) + "\n"
                    light.set_brightness(min(light.brightness, brightness))

        elif Type == "Toilet":
            for light in byrooms[room]:
                print "c"  # print light.brightness
                if ACTUATE:
                    brightness = 10
                    Actuated += "Lights in room" + str(room) + " was set to" + str(brightness) + "\n"
                    light.set_brightness(min(light.brightness, brightness))
        else:
            query_list = []
            for i in byrooms_o[room]:
                query_list.append(i)

            # get the sensor data
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))
            dfs = c.do_query({'Composition': query_list,
                              'Selectors': [mdal.MAX] * len(query_list),
                              'Time': {'T0': (now - timedelta(hours=0.5)).strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                       'T1': now.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                                       'WindowSize': str(30) + 'min',
                                       'Aligned': True}})

            dfs = pd.concat([dframe for uid, dframe in dfs.items()], axis=1)

            df = dfs[[query_list[0]]]
            df.columns.values[0] = 'occ'
            df.is_copy = False
            df.columns = ['occ']
            # perform OR on the data, if one sensor is activated, the whole room is considered occupied
            for i in range(1, len(query_list)):
                df.loc[:, 'occ'] += dfs[query_list[i]]
            df.loc[:, 'occ'] = 1 * (df['occ'] > 0)

            if df["occ"][0] != 1:
                for light in byrooms[room]:
                    if ACTUATE:
                        brightness = 0
                        Actuated += "Lights in room" + str(room) + " was set to " + str(brightness) + "\n"
                        light.set_brightness(min(light.brightness, brightness))
            else:
                for light in byrooms[room]:
                    print light.brightness
                    if ACTUATE:
                        brightness = 10
                        Actuated += "Lights in room" + str(room) + " was set to " + str(brightness) + "\n"
                        light.set_brightness(min(light.brightness, brightness))

    print "Done!"
    print "================================"
    print "We skipped the following lights:"
    print skipped
    print "We actuated the following lights:"
    print Actuated
