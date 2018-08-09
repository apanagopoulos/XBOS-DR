import sys
import threading
import pytz
import datetime

sys.path.append("./MPC")

import utils
from Advise import Advise
from DataManager import DataManager
from ThermalDataManager import ThermalDataManager

# TODO update this whole advise to use the newest data methods

class IteratedDP:
    def __init__(self, building, client, num_threads, num_iterations=1):
        """
        :param building: str building name.
        :param client: Client object.
        :param num_threads: int. number of threads 
        """
        # Set building
        self.building = building
        # Set client
        self.client = client

        # To make threads wait together for next iteration
        self.barrier = utils.Barrier(num_threads)
        # To access shared variables
        self.mutex = threading.Semaphore(1)

        # Set number of iterations to run in iterate_advise
        self.num_iterations = num_iterations




    def advise(self, start, end, zones, cfg_building, cfg_zones, thermal_models,
                    zone_temperatures, consumption_storage):
        """
        
        :param start: 
        :param end: 
        :param zones:
        :param cfg_building: 
        :param cfg_zones: 
        :param thermal_model: 
        :param zone_temperatures: 
        :param consumption_storage: 
        :return: 
        """

        cfg_zone = cfg_zones[zones[0]]
        # TODO fix end
        start_utc = start.astimezone(tz=pytz.utc)
        end_utc = start_utc + datetime.timedelta(hours=cfg_zone["Advise"]["MPCPredictiveHorizon"])
        # end_utc = end.astimezone(tz=pytz.utc)

        start_cfg_timezone = start_utc.astimezone(tz=utils.get_config_timezone(cfg_building))
        end_cfg_timezone = end_utc.astimezone(tz=utils.get_config_timezone(cfg_building))



        # get datamanagers
        # TODO Set up multizones.
        data_manager = DataManager(cfg_building, cfg_zone, self.client, zones[0], now=start_utc)
        thermal_data_manager = ThermalDataManager(cfg_building, self.client)

        safety_constraints = data_manager.safety_constraints()
        prices = data_manager.prices()
        building_setpoints = data_manager.building_setpoints()

        # TODO implement correct occupancy
        occ_predictions = data_manager.preprocess_occ_cfg()

        # Getting outside temperatures.
        outside_temperatures = utils.get_outside_temperatures(cfg_building, cfg_zone, start_utc, end_utc,
                                                              data_manager, thermal_data_manager,
                                                              interval=cfg_building["Interval_Length"])

        # For thermal model need to set weather predictions for every loop and set current zone temperatures.
        for zone, zone_thermal_model in thermal_models.items():
            zone_thermal_model.set_temperatures(zone_temperatures[zone])
            zone_thermal_model.set_outside_temperature(outside_temperatures)

        is_DR = utils.is_DR(start_utc, cfg_building)

        for i in range(self.num_iterations):
            self.barrier.wait()

            adv = Advise(zones,  # array because we might use more than one zone. Multiclass approach.
                         start_cfg_timezone,
                         occ_predictions,
                         [zone_temperatures[zones[0]]],
                         thermal_models,
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
                         False, # TODO cfg_zone["Advise"]["Occupancy_Sensors"] if not simulate else False,
                         # TODO Only using config file occupancy for now.
                         safety_constraints)

            action = adv.advise()

            # # TODO make adv.get_consumption return a dictionary of consumption for each zone.
            # zone_consumption = adv.get_consumption()
            # consumption_storage.update_zone_consumption(zones, {zones[0]: zone_consumption})

        # return the last action we got
        return action
