import sys
import threading

sys.path.append("./MPC")

import utils
import Advise


class IteratedDP:
    def __init__(self, num_threads, num_iterations=5):
        """
        
        :param num_threads: int. number of threads 
        """
        # To make threads wait together for next iteration
        self.barrier = utils.Barrier(num_threads)
        # To access shared variables
        self.mutex = threading.Semaphore(1)

        # Set number of iterations to run in iterate_advise
        self.num_iterations = num_iterations


    def iterate_advise(self, start, end, num_iterations, zones, cfg_building, cfg_zone, thermal_model,
                       building_setpoints, safety_constraints, prices, occ_predictions, zone_temperatures, is_DR,
                       consumption_storage, simulate=True):
        """
        
        :param start: 
        :param end: 
        :param num_iterations: 
        :param zones: 
        :param cfg_building: 
        :param cfg_zone: 
        :param thermal_model: 
        :param building_setpoints: 
        :param safety_constraints: 
        :param prices: 
        :param occ_predictions: 
        :param zone_temperatures: 
        :param is_DR: 
        :param simulate: 
        :return: 
        """
        for i in range(self.num_iterations):
            self.barrier.wait()

            adv = Advise(zones,  # array because we might use more than one zone. Multiclass approach.
                         start,
                         occ_predictions,
                         zone_temperatures,
                         thermal_model,
                         prices,
                         cfg_zone["Advise"]["General_Lambda"],
                         cfg_zone["Advise"]["DR_Lambda"],
                         is_DR,
                         cfg_building["Interval_Length"],
                         cfg_zone["Advise"]["MPCPredictiveHorizon"],
                         cfg_zone["Advise"]["Heating_Consumption"],
                         cfg_zone["Advise"]["Cooling_Consumption"],
                         cfg_zone["Advise"]["Ventilation_Consumption"],
                         cfg_zone["Advise"]["Thermal_Precision"],
                         cfg_zone["Advise"]["Occupancy_Obs_Len_Addition"],
                         building_setpoints,
                         cfg_zone["Advise"]["Occupancy_Sensors"] if not simulate else False,
                         # TODO Only using config file occupancy for now.
                         safety_constraints)

            action = adv.advise()
            # TODO make adv.get_consumption return a dictionary of consumption for each zone.
            zone_consumption = adv.get_consumption()
            consumption_storage.update_zone_consumption(zones, {zones[0]: zone_consumption})
