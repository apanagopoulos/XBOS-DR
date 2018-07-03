#this is the plotter for the MPC graph

import pandas as pd
import numpy as np
import datetime
import pytz

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
Utility functions
'''
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


# Heating
def f1(row):
    """
    helper function to format the thermal model dataframe
    """
    if row['action'] == 1.:
        val = 1
    else:
        val = 0
    return val


# if state is 2 we are doing cooling
def f2(row):
    """
    helper function to format the thermal model dataframe
    """
    if row['action'] == 2.:
        val = 1
    else:
        val = 0
    return val


def f3(row):
    """
    helper function to format the thermal model dataframe
    """
    if 0 < row['a'] <= 1:
        return 1
    elif 1 < row['a'] <= 2:
        return 2
    elif np.isnan(row['a']):
        return row['a']
    else:
        return 0

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

        tstat.write({"override": False})
        import os
        if not os.path.exists("Buildings/" + building + "/Logs"):
            os.makedirs("Buildings/" + building + "/Logs")

        if os.path.exists("Buildings/" + building + "/Logs/" + zone + ".log"):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        logfile = open("Buildings/" + building + "/Logs/" + zone + ".log", append_write)
        logfile.write(
            "THERMOSTAT CHANGED MANUALY AT : " + datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC")
        logfile.close()
    return flag_changed


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

if __name__ == '__main__':
    print get_utc_now()

    # ('%Y-%m-%d %H:%M:%S') + ' UTC'

    s = get_mdal_string_to_datetime("2018-09-01 12:32:12 UTC")

    e = get_mdal_string_to_datetime("2018-09-10 21:12:51 UTC")

    d = e - s
    print(d)
    print(d/2)


