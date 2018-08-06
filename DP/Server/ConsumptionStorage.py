import utils

import pandas as pd
import numpy as np
import datetime
import threading
import time

class ConsumptionStorage:
    """A wrapper class for consumption data storage which is shared by all zone threads. This will 
    be used for the demand charges for knowing the consumption in the building and for every zone. Will be useful
    when iterating zone optimization to converge at an optimal solution for the demand charges. 
    
    After every iteration, once all zones have optimized, each zone will update their consumption which will be used
    by all other zones in the next iteration."""

    def __init__(self, building, num_threads, building_consumption=None):
        """
        
        :param building_consumption: pd.Series
        :param building: 
        :param num_threads: (int) the number of threads that will use this class. 
        """
        self.building = building
        self.zones = utils.get_zones(building)

        # the barrier variable needed to make sure all zones update their consumption at the same time.
        self.barrier = utils.Barrier(num_threads)
        # make sure that only one thread at a time can modify shared data.
        self.mutex = threading.Semaphore(1)

        # initializing variables and setting them with update_building_consumption
        self.building_consumption = None
        self.zone_consumption = None
        if building_consumption is not None:
            self.update_building_consumption(building_consumption)



    def reset_zones(self):
        """Resets the zone consumption to all zeroes."""
        for zone in self.zones:
            self.zone_consumption[zone] = np.zeros(self.zone_consumption.shape[0])

    # TODO make sure only one thread is modifying
    def update_building_consumption(self, building_consumption):
        """Upates the building consumption and sets new zone consumption.
        :param building_consumption: pd.Series with timeseries that will be used for the zones.
        """

        self.mutex.acquire()
        # set building consumption
        self.building_consumption = building_consumption

        self.zone_consumption = pd.DataFrame(columns=self.zones, index=self.building_consumption.index)
        self.reset_zones()
        self.mutex.release()

    def update_zone_consumption(self, zones, zone_data, debug=False):
        """
        Updates the zone consumption. Needs to be called after every iteration for the demand charges. 
        :param zones: list of strings 
        :param zone_data: {zone: data} data is pd.df or pd.series with same index as the set building consumption.
        :return: 
        """
        # waiting for all threads to wait here
        if debug:
            # if zones[0] == "HVAC_Zone_Eastzone":
            #     time.sleep(10)
            print("%s got to barrier." % zones)
        self.barrier.wait()
        if debug:
            print("%s got past barrier." % zones)

        # setting the consumption data.
        # Need to make sure that we have the same index as the building consumption.
        for zone in zones:
            assert zone_data[zone].index.equals(self.building_consumption.index)
            self.mutex.acquire()
            self.zone_consumption[zone] = zone_data[zone].values
            self.mutex.release()

    def get_rest_of_building_consumption(self, zones, consumption_time):
        """
        Gets the consumption of the building, minus the stored consumption for the given zones.
        :param zones: (list str) the list of zones which should be excluded from the zone consumption.
        :param consumption_time: datetime in timezone of local building time.
        :return: float. The consumption of the building without the given zones. 
        """
        rest_consumption = self.building_consumption.loc[consumption_time]
        for temp_zone in self.zones:
            if temp_zone not in zones:
                rest_consumption += self.zone_consumption.loc[consumption_time, temp_zone]
        return rest_consumption



if __name__ == "__main__":
    end = utils.get_utc_now()
    BUILDING = "CIEE"
    ZONES = utils.get_zones(BUILDING)
    building_consumption = pd.Series(data=[10, 20, 30], index=[end - datetime.timedelta(minutes=30)
                                                               , end - datetime.timedelta(minutes=15),
                                                               end])

    zone_data = pd.Series(data=[10, 20, 30], index=[end - datetime.timedelta(minutes=30)
                                                               , end - datetime.timedelta(minutes=15),
                                                               end])

    consumption_storage = ConsumptionStorage(BUILDING, 4, building_consumption)

    print("Updating consumption.")
    # setting up the threads to call the update function
    class conThread(threading.Thread):
        def __init__(self, zones, zone_data, consumption_storage):
            threading.Thread.__init__(self)

            self.zones = zones
            self.zone_data = zone_data
            self.consumption_storage = consumption_storage

        def run(self):
            self.consumption_storage.update_zone_consumption(self.zones, self.zone_data, debug=True)


    threads = []
    print(ZONES)
    for zone in ZONES:
        threads.append(conThread([zone], {zone: zone_data}, consumption_storage))
        threads[-1].start()  # Starts the thread.
    for thread in threads:
        """Waits for the threads to complete before moving on
           with the main script.
        """
        thread.join()
    print("Threads All done.")

    print(consumption_storage.get_consumption_of_rest([ZONES[0], ZONES[1]], end))