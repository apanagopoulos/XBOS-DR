#this is the plotter for the MPC graph

import pandas as pd
import numpy as np
import datetime
import pytz
import yaml
import os
import string

# be careful of circular import.
# https://stackoverflow.com/questions/11698530/two-python-modules-require-each-others-contents-can-that-work
import ThermalDataManager


from xbos import get_client

from xbos.devices.thermostat import Thermostat

import plotly.graph_objs as go

try:
	import pygraphviz
	from networkx.drawing.nx_agraph import graphviz_layout
	#print("using package pygraphviz")
except ImportError:
	try:
		import pydotplus
		from networkx.drawing.nx_pydot import graphviz_layout
		#print("using package pydotplus")
	except ImportError:
		print()
		print("Both pygraphviz and pydotplus were not found ")
		print("see http://networkx.github.io/documentation"
			  "/latest/reference/drawing.html for info")
		print()
		raise

'''
Utility constants
'''
NO_ACTION = 0
HEATING_ACTION = 1
COOLING_ACTION = 2
FAN = 3
TWO_STAGE_HEATING_ACTION = 4
TWO_STAGE_COOLING_ACTION = 5

SERVER_DIR_PATH = UTILS_FILE_PATH = os.path.dirname(__file__) # this is true for now

'''
Utility functions
'''
# ============ BUILDING AND ZONE GETTER ========
def choose_building_and_zone():
    print "-----------------------------------"
    print "Buildings:"
    print "-----------------------------------"
    root, dirs, files = os.walk(SERVER_DIR_PATH + "/Buildings/").next()
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

# ============ DATE FUNCTIONS ============

def get_utc_now():
    """Gets current time in utc time.
    :return Datetime in utctime zone"""
    return datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC"))

def in_between(now, start, end):
    """Finds whether now is between start and end. Takes care of cases such as start=11:00pm and end=1:00am 
    now = 00:01, and hence would return True. 
    :param now: (datetime.time) 
    :param start: (datetime.time) 
    :param end: (datetime.time) 
    :return (boolean)"""
    if start < end:
        return start <= now < end
    # when end is in the next day.
    elif end < start:
        return start <= now or now < end
    else:
        return True


def get_time_datetime(time_string):
    """Gets datetime from string with format HH:MM.
    :param date_string: string of format HH:MM
    :returns datetime.time() object with no associated timzone. """
    return datetime.datetime.strptime(time_string, "%H:%M").time()


def get_mdal_string_to_datetime(date_string):
    """Gets datetime from string with format Year-Month-Day Hour:Minute:Second UTC. Note, string should be for utc
    time.
    :param date_string: string of format Year-Month-Day Hour:Minute:Second UTC.
    :returns datetime.time() object in UTC time. """
    return datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=pytz.timezone("UTC"))

def get_mdal_datetime_to_string(date_object):
    """Gets string from datetime object. In UTC Time.
    :param date_object
    :returns '%Y-%m-%d %H:%M:%S UTC' """
    return date_object.strftime('%Y-%m-%d %H:%M:%S') + ' UTC'


# ============ DATA FUNCTIONS ============

def round_increment(data, precision=0.05):
    """Round to nearest increment of precision.
    :param data: np.array of floats or single float
    :param precision: (float) the increment to round to
    :return (np.array or float) of rounded floats."""
    # source for rounding: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    return precision * np.round(data / float(precision))

def is_cooling(action_data):
    """Returns boolen area of actions which were cooling (either two or single stage).
    :param action_data: np.array or pd.series"""
    return (action_data == COOLING_ACTION) | (action_data == TWO_STAGE_COOLING_ACTION)


def is_heating(action_data):
    """Returns boolen area of actions which were heating (either two or single stage).
    :param action_data: np.array or pd.series"""
    return (action_data == HEATING_ACTION) | (action_data == TWO_STAGE_HEATING_ACTION)


def choose_client(cfg):
    if cfg["Server"]:
        client = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        client = get_client()
    return client


def get_config(building):

    config_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + building + ".yml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.load(f)
    except:
        print("ERROR: No config file for building %s with path %s" % (building, config_path))
        return
    return cfg


def get_zone_config(building, zone):
    config_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + "ZoneConfigs/" + zone + ".yml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.load(f)
    except:
        print("ERROR: No config file for building %s and zone % s with path %s" % (building, zone, config_path))
        return
    return cfg

def get_zone_log(building, zone):


    log_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + "Logs/" + zone + ".log"

	## fix for one lines
    try:

		f = open (log_path, "r")
		log=f.read()
		log = string.replace(log, "UTCTHERMOSTAT", "UTC\nTHERMOSTAT")
		f.close()

		f = open(log_path, 'w')
		f.write(log)
		f.close()
    except:
        print("ERROR: No config file for building %s and zone % s with path %s" % (building, zone, log_path))
        return
	## end of fix DELETE THIS WHEN ALL LOGS ARE FIXED!

    try:
        with open (log_path, "r") as f:
			### fix for same line logs ###
            log=f.readlines()
    except:
        print("ERROR: No config file for building %s and zone % s with path %s" % (building, zone, log_path))
        return
    return log


# Maybe put in ThermalDataManager because of circular import.
def get_data(building=None, client=None, cfg=None, start=None, end=None, days_back=50, evaluate_preprocess=False, force_reload=False):
    """
    Get preprocessed data.
    :param building: (str) building name
    :param cfg: (dictionary) config file for building. If none, the method will try to find it. 
    :param days_back: how many days back from current moment.
    :param evaluate_preprocess: (Boolean) should controller data manager add more features to data.
    :param force_reload: (boolean) If some data for this building is stored, the reload if not force reload. Otherwise,
                        load data as specified.
    :param start: the start time for the data. If none is given, we will use days_back to go back from the
                     end datetime if given (end - days_back), or the current time.
    :param end: the end time for the data. If given, we will use it as our end. If not given, we will use the current 
                    time as the end.
    :return: {zone: pd.df with columns according to evaluate_preprocess}
    """
    assert cfg is not None or building is not None
    if cfg is not None:
        building = cfg["Building"]
    else:
        cfg = get_config(building)

    print("----- Get data for Building: %s -----" % building)

    if evaluate_preprocess:
        path = SERVER_DIR_PATH + "/Thermal_Data/" + building + "_eval"
    else:
        path = SERVER_DIR_PATH + "/Thermal_Data/" + building

    if end is None:
        end = get_utc_now()
    if start is None:
        start = end - datetime.timedelta(days=days_back)

    # TODO ugly try/except
    try:
        assert not force_reload
        print(path)
        with open(path, "r") as f:
            import pickle
            thermal_data = pickle.load(f)
    except:
        if client is None:
            client = choose_client(cfg)
        dataManager = ThermalDataManager.ThermalDataManager(cfg, client)
        thermal_data = dataManager.thermal_data(start=start, end=end, evaluate_preprocess=evaluate_preprocess)
        with open(path, "wb") as f:
            import pickle
            pickle.dump(thermal_data, f)
    return thermal_data


def get_raw_data(building=None, client=None, cfg=None, start=None, end=None, days_back=50, force_reload=False):
    assert cfg is not None or building is not None
    if cfg is not None:
        building = cfg["Building"]
    else:
        config_path = SERVER_DIR_PATH + "/Buildings/" + building + "/" + building + ".yml"
        try:
            with open(config_path, "r") as f:
                cfg = yaml.load(f)
        except:
            print("ERROR: No config file for building %s with path %s" % (building, config_path))
            return

    print("----- Get data for Building: %s -----" % building)

    path = SERVER_DIR_PATH + "/Thermal_Data/" + building
    # TODO ugly try/except

    if end is None:
        end = get_utc_now()
    if start is None:
        start = end - datetime.timedelta(days=days_back)


    # inside and outside data data
    import pickle
    try:
        assert not force_reload
        with open(path + "_inside", "r") as f:
            inside_data = pickle.load(f)
        with open(path + "_outside", "r") as f:
            outside_data = pickle.load(f)
    except:
        if client is None:
            client = get_client()
        dataManager = ThermalDataManager.ThermalDataManager(cfg, client)

        inside_data = dataManager._get_inside_data(start, end)
        outside_data = dataManager._get_outside_data(start, end)
        with open(path + "_inside", "wb") as f:
            pickle.dump(inside_data, f)
        with open(path + "_outside", "wb") as f:
            pickle.dump(outside_data, f)
    return inside_data, outside_data


def get_mdal_data(mdal_client, query):
    """Gets mdal data. Necessary method because if a too long time frame is queried, mdal does not return the data.
    :param mdal_client: mdal object to query data.
    :param query: mdal query
    :return pd.df with composition as columns. Timeseries in UTC time."""
    start = get_mdal_string_to_datetime(query["Time"]["T0"])
    end = get_mdal_string_to_datetime(query["Time"]["T1"])
    time_frame = end - start

    # get windowsize
    str_window = query["Time"]["WindowSize"]
    assert str_window[-3:] == "min"
    WINDOW_SIZE = datetime.timedelta(minutes=int(str_window[:-3]))

    if time_frame < WINDOW_SIZE:
        raise Exception("WindowSize is less than the time interval for which data is requested.")

    # To get logarithmic runtime we take splits which are powers of two.
    max_interval = datetime.timedelta(hours=12)  # the maximum interval length in which to split the data.
    max_num_splits = int(time_frame.total_seconds()//max_interval.total_seconds())
    all_splits = [1]
    for _ in range(2, max_num_splits):
        power_split = all_splits[-1] * 2
        if power_split > max_num_splits:
            break
        all_splits.append(power_split)

    received_all_data = False
    outside_data = []
    # start loop to get data in time intervals of logarithmically decreasing size. This will hopefully find the
    # spot at which mdal returns data.
    for num_splits in all_splits:
        outside_data = []
        pre_look_ahead = time_frame / num_splits

        # to round down to nearest window size multiple
        num_window_in_pre_look = pre_look_ahead.total_seconds()//WINDOW_SIZE.total_seconds()
        look_ahead = datetime.timedelta(seconds=WINDOW_SIZE.total_seconds() * num_window_in_pre_look)

        print("Attempting to get data in %f day intervals." % (look_ahead.total_seconds()/(60*60*24)))

        temp_start = start
        temp_end = temp_start + look_ahead

        while temp_end <= end:
            query["Time"]["T0"] = get_mdal_datetime_to_string(temp_start)
            query["Time"]["T1"] = get_mdal_datetime_to_string(temp_end)
            mdal_outside_data = mdal_client.do_query(query, tz="UTC")
            if mdal_outside_data == {}:
                print("Attempt failed.")
                received_all_data = False
                break
            else:
                outside_data.append(mdal_outside_data["df"])

                # advance temp_start and temp_end
                temp_start = temp_end + WINDOW_SIZE
                temp_end = temp_start + look_ahead

                # to get rest of data if look_ahead is not exact mutliple of time_between
                if temp_start < end < temp_end:
                    temp_end = end

                # To know that we received all data.
                if end < temp_start:
                    received_all_data = True

        # stop if we got the data
        if received_all_data:
            print("Succeeded.")
            break


    if not received_all_data:
        raise Exception("WARNING: Unable to get data form MDAL.")

    return pd.concat(outside_data)



def concat_zone_data(thermal_data):
    """Concatinates all thermal data zone data into one big dataframe. Will sort by index. Get rid of all zone_temperature columns.
    :param thermal_data: {zone: pd.df}
    :return pd.df without zone_temperature columns"""
    concat_data = pd.concat(thermal_data.values()).sort_index()
    filter_columns = ["zone_temperature" not in col for col in concat_data.columns]
    return concat_data[concat_data.columns[filter_columns]]


def as_pandas(result):
    time = result[list(result.keys())[0]][:, 0]
    df = pd.DataFrame(time, columns = ['Time'])
    df['Time'] = pd.to_datetime(df['Time'], unit='s')


    for key in result:
        df[key] = result[key][:, 1].tolist()
        try:
            df[key + " Var"] = result[key][:, 2].tolist()
        except IndexError:
            pass

    df = df.set_index('Time')
    return df


# ============ THERMOSTAT FUNCTIONS ============


def has_setpoint_changed(tstat, setpoint_data, zone, building):
    """
    Checks if thermostats was manually changed and prints warning.
    :param tstat: Tstat object we want to look at.
    :param setpoint_data: dict which has keys {"heating_setpoint": bool, "cooling_setpoint": bool} and corresponds to
            the setpoint written to the thermostat by MPC.
    :param zone: Name of the zone to print correct messages.
    :return: Bool. Whether tstat setpoints are equal to setpoints written to tstat.
    """
    WARNING_MSG = "WARNING. %s has been manually changed in zone %s. Setpoint is at %s from expected %s. " \
                  "Setting override to False and intiatiating program stop."
    flag_changed = False
    if tstat.cooling_setpoint != setpoint_data["cooling_setpoint"]:
        flag_changed = True
        print(WARNING_MSG % ("cooling setpoint", zone, tstat.cooling_setpoint, setpoint_data["cooling_setpoint"]))
    if tstat.heating_setpoint != setpoint_data["heating_setpoint"]:
        flag_changed = True
        print(WARNING_MSG % ("heating setpoint", zone, tstat.heating_setpoint, setpoint_data["heating_setpoint"]))

    # write override false so the local schedules can take over again.
    if flag_changed:

        set_override_false(tstat)
        import os
        if not os.path.exists("Buildings/" + building + "/Logs"):
            os.makedirs("Buildings/" + building + "/Logs")

        if os.path.exists("Buildings/" + building + "/Logs/" + zone + ".log"):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        logfile = open("Buildings/" + building + "/Logs/" + zone + ".log", append_write)
        logfile.write(
            "THERMOSTAT CHANGED MANUALY AT : " + datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC \n")
        logfile.close()
    return flag_changed


def set_override_false(tstat):
    tstat.write({"override": False})


def get_thermostats(client, hod, building):
    """Gets the thermostats for given building.
    :param client: xbos client object
    :param hod: hod client object
    :param building: (string) building name
    :return {zone: tstat object}"""

    query = """SELECT ?uri ?zone FROM %s WHERE {
        ?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
        ?tstat bf:uri ?uri .
        ?tstat bf:controls/bf:feeds ?zone .
        };"""

    # Start of FIX for missing Brick query
    query = """SELECT ?zone ?uri FROM  %s WHERE {
              ?tstat rdf:type brick:Thermostat .
              ?tstat bf:controls ?RTU .
              ?RTU rdf:type brick:RTU .
              ?RTU bf:feeds ?zone. 
              ?zone rdf:type brick:HVAC_Zone .
              ?tstat bf:uri ?uri.
              };"""
    # End of FIX - delete when Brick is fixed
    building_query = query % building

    tstat_query_data = hod.do_query(building_query)['Rows']
    tstats = {tstat["?zone"]: Thermostat(client, tstat["?uri"]) for tstat in tstat_query_data}
    return tstats

# ============ PLOTTING FUNCTIONS ============


def plotly_figure(G, path=None):

    pos = graphviz_layout(G, prog='dot')

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    my_annotations = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        my_annotations.append(
            dict(
                x=(x0+x1)/2,
                y=(y0+y1)/2,
                xref='x',
                yref='y',
                text=G.get_edge_data(edge[0], edge[1])['action'], # TODO for multigraph use [0] to get the frist edge.
                showarrow=False,
                arrowhead=2,
                ax=0,
                ay=0
            )
        )


    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.Marker(
            showscale=False,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'].append(x)
        node_trace['y'].append(y)

        node_info = "Time: +{0}<br>Temps: {1}<br>Usage Cost: {2}".format(node.time,
                                                                           node.temps,
                                                                           G.node[node]['usage_cost'])

        node_trace['text'].append(node_info)

        if path is None:
            node_trace['marker']['color'].append(G.node[node]['usage_cost'])
        elif node in path:
            node_trace['marker']['color'].append('rgba(255, 0, 0, 1)')
        else:
            node_trace['marker']['color'].append('rgba(0, 0, 255, 1)')


    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont=dict(size=16),
                        showlegend=False,
                        width=650,
                        height=650,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=my_annotations,
                        xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

if __name__ == "__main__":
    # bldg = "csu-dominguez-hills"
    # inside, outside = get_raw_data(building=bldg, days_back=20, force_reload=True)
    # use_data = {}
    # for zone, zone_data in inside.items():
    #     if zone != "HVAC_Zone_Please_Delete_Me":
    #         use_data[zone] = zone_data
    #         print(zone)
    #         print(zone_data[zone_data["action"] == 2].shape)
    #         print(zone_data[zone_data["action"] == 5].shape)
    #
    # t_man = ThermalDataManager.ThermalDataManager({"Building": bldg}, client=get_client())
    # outside = t_man._preprocess_outside_data(outside.values())
    # print("inside")
    # th_data = t_man._preprocess_thermal_data(use_data, outside, True)


    import pickle
    with open("u_p", "r") as f:
        th = pickle.load(f)

    zone = "HVAC_Zone_SAC_2101"
    zone_data = th[zone]
    print(zone_data[zone_data["action"] == 5].shape)
    print(zone_data[zone_data["action"] == 2].shape)