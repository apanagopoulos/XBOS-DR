
from xbos import get_client
from xbos.services.hod import HodClientHTTP
from xbos.devices.thermostat import Thermostat

client = get_client(agent = '172.17.0.1:28589', entity="thanos.ent")
hc = HodClientHTTP("http://ciee.cal-sdb.org")

q = """SELECT ?uri ?zone WHERE {
    ?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
    ?tstat bf:uri ?uri .
    ?tstat bf:controls/bf:feeds ?zone .
};
"""

zones = {}
for tstat in hc.do_query(q):
    print tstat
    zones[tstat["?zone"]] = Thermostat(client, tstat["?uri"])

normal_zones = ["EastZone"]
p = {"override": True, "heating_setpoint": 62., "cooling_setpoint": 85.}
for z in normal_zones:
    zones[z].write(p)
