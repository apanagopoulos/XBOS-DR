#this is the plotter for the MPC graph

import pandas as pd
import datetime

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
# ============ DATE FORMATTING ============


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


def get_datetime(date_string):
    """Gets datetime from string with format HH:MM.
    :param date_string: string of format HH:MM
    :returns datetime.time() object with no associated timzone. """
    return datetime.datetime.strptime(date_string, "%H:%M").time()

# ============ DATA FUNCTIONS ============


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


def has_setpoint_changed(tstat, setpoint_data, zone):
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
                text=G.get_edge_data(edge[0], edge[1])['action'],
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
