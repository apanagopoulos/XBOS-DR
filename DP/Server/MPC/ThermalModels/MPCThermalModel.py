import sys

import numpy as np
import pandas as pd
import yaml

# daniel imports
sys.path.append("./ThermalModels")
sys.path.append("..")

import utils
from ThermalModel import ThermalModel
from LinearThermalModel import LinearThermalModel


class MPCThermalModel:
    """Class specifically designed for the MPC process. A child class of ThermalModel with functions
        designed to simplify usage.
        
        NOTE: when predicting will always round."""

    def __init__(self, zone, thermal_data, interval_thermal, interval_length, thermal_precision=0.05, is_two_stage=False):
        """
        :param zone: The zone this Thermal model is meant for. 
        :param thermal_data: pd.df thermal data for zone (as preprocessed by ControllerDataManager). Only used for fitting.
        :param interval_length: (int) Number of minutes between controls.
        :param thermal_precision: (float) The increment to which to round predictions to. (e.g. 1.77 becomes 1.75
        :param is_two_stage: (bool) Whether this for a two stage thermal model. Important for consistency check.
         and 4.124 becomes 4.10)
        """
        self.zone = zone
        self.is_two_stage = is_two_stage

        # set the thermal models to be used
        self.thermal_model = ThermalModel(interval_thermal=interval_thermal, thermal_precision=thermal_precision)
        self.linear_thermal_model = LinearThermalModel(interval_thermal=interval_thermal, thermal_precision=thermal_precision)


        # fit the thermal models
        self.thermal_model = self.thermal_model.fit(thermal_data, thermal_data["t_next"])
        self.linear_thermal_model = self.linear_thermal_model.fit(thermal_data, thermal_data["t_next"])

        # online training debugging.
        self._oldParams = {}

        # in case we want to set it during creation and don't have access during prediction.
        self.interval = interval_length
        self.interval_thermal = interval_thermal

        # the data that will be stored through the MPCThermal Model to make usage easier.
        self.zone_temperatures = None
        self.outside_temperature = None  # hour corresponds to PST hour of the day used.
        self.last_action = None

    def set_last_action(self, action):
        """Set the action that was last used by MPC for later online training.
        :param action: (int) action nr as given in utils."""
        self.last_action = action

    def set_outside_temperature(self, outside_temperature):
        """Set weather predictions after last fit to store for later predictions.
        :param outside_temperature: (float []) 0th index corresponds to t_out now and 1st index to the temperature 
        outside in one hour from now, etc. Now is defined as the start time of the advise graph."""
        self.outside_temperature = outside_temperature

    def _datapoint_to_dataframe(self, interval, t_in, action, t_out, zone_temperatures):
        """A helper function that converts a datapoint to a pd.df used for predictions.
        Assumes that we have self.zone"""
        X = {"dt": interval, "action": action,
             "t_out": t_out}
        for key_zone, val in zone_temperatures.items():
            if key_zone != self.zone:
                X["zone_temperature_" + key_zone] = val
            else:
                X["t_in"] = t_in

        return pd.DataFrame(X, index=[0], dtype=float)

    def set_temperatures(self, curr_zone_temperatures, interval=15):
        """
        Stores curr temperature for every zone. Call whenever we are starting new interval.
        :param curr_zone_temperatures: {zone: temperature}
        :return: None
        """
        # set new zone temperatures.
        self.zone_temperatures = curr_zone_temperatures
        return

    def update(self, curr_zone_temperatures, interval=15):

        """Performs one update step for the thermal model. Call at the beginning of each interval once we know the
        new temperatures but before we set them.
        :param curr_zone_temperatures: {zone: temperature}
        :param interval: The delta time since the last action was called. 
        :return: None
        """
        # TODO Fix this to get online learning going.
        pass

    def predict(self, t_in, action, outside_temperature, interval,
                max_heating_per_minute=0.5, max_cooling_per_minute=0.5, debug=False):
        """
        Predicts temperature for zone given.
        :param t_in: float
        :param action: (float)
        :param outside_temperature: float
        :param interval: float
        :param debug: whether to debug, meaning return the type of thermal model that causes each prediction.
        :return: not debug: (np.array) predictions in order. 
                 debug: (np.array) predictions in order, (np.array dtype=object/strings) thermal_model types 
        """
        if interval is None:
            interval = self.interval

        t_out = outside_temperature

        # Get predictions for all actions to accuratly do consistency checks. NOTE: Not doing 2 stage here.
        X_heating = self._datapoint_to_dataframe(interval, t_in, utils.HEATING_ACTION, t_out, self.zone_temperatures)
        X_cooling = self._datapoint_to_dataframe(interval, t_in, utils.COOLING_ACTION, t_out, self.zone_temperatures)
        X_no_action = self._datapoint_to_dataframe(interval, t_in, utils.NO_ACTION, t_out, self.zone_temperatures)

        prediction_heating = self.thermal_model.predict(X_heating)[0]
        prediction_cooling = self.thermal_model.predict(X_cooling)[0]
        prediction_no_action = self.thermal_model.predict(X_no_action)[0]

        # Consistency and sensibility checks for all predictions.
        prediction_heating = max(prediction_heating, t_in + self.thermal_model.thermal_precision)
        prediction_heating = min(prediction_heating, t_in + max_heating_per_minute * interval)

        prediction_cooling = min(prediction_cooling, t_in - self.thermal_model.thermal_precision)
        prediction_cooling = max(prediction_cooling, t_in + max_cooling_per_minute * interval)

        prediction_no_action = max(prediction_no_action, prediction_cooling)
        prediction_no_action = min(prediction_no_action, prediction_heating)

        if utils.is_heating(action):
            return prediction_heating
        elif utils.is_cooling(action):
            return prediction_cooling
        else:
            return prediction_no_action

    def save_to_config(self):
        # this does not work anymore as intended.
        return

        """saves the whole model to a yaml file.
        RECOMMENDED: PYAML should be installed for prettier config file."""
        config_dict = {}

        # store zone temperatures
        config_dict["Zone Temperatures"] = self.zone_temperatures

        # store coefficients
        coefficients = {parameter_name: param for parameter_name, param in
                        zip(super(MPCThermalModel, self)._params_order, super(MPCThermalModel, self)._params)}
        config_dict["coefficients"] = coefficients

        # store evaluations and RMSE's.
        config_dict["Evaluations"] = {}
        config_dict["Evaluations"]["Baseline"] = super(MPCThermalModel, self).baseline_error
        config_dict["Evaluations"]["Model"] = super(MPCThermalModel, self).model_error
        config_dict["Evaluations"]["ActionOrder"] = super(MPCThermalModel, self).scoreTypeList
        config_dict["Evaluations"]["Better Than Baseline"] = super(MPCThermalModel, self).betterThanBaseline

        with open("../ZoneConfigs/thermal_model_" + self.zone, 'wb') as ymlfile:
            # TODO Note import pyaml here to get a pretty config file.
            try:
                import pyaml
                pyaml.dump(config_dict[self.zone], ymlfile)
            except ImportError:
                yaml.dump(config_dict[self.zone], ymlfile)


if __name__ == "__main__":
    choose_buildings = False
    if choose_buildings:
        building, zone = utils.choose_building_and_zone()
    else:
        building, zone = "avenal-veterans-hall", "HVAC_Zone_AC-4"

    data = utils.get_data(building, force_reload=False, evaluate_preprocess=False, days_back=150)
    zone_data = data[zone]
    zone_data = zone_data[zone_data['dt'] == 5]

    # get zone temperatures
    zone_temps = {}
    for zone_key in data.keys():
        zone_temps[zone_key] = 70

    print(zone_data.index)

    mpc_thermal_model = MPCThermalModel(zone, zone_data, 5, is_two_stage=False)

    mpc_thermal_model.set_temperatures_and_fit(zone_temps, 0)
    # print(zone_data[0 == mpc_thermal_model._consistency_test(zone_data, mpc_thermal_model.thermal_model)].values[0])
    # print(mpc_thermal_model.thermal_model._params)
    # print(mpc_thermal_model.predict(zone_data))
    print(mpc_thermal_model.predict(70, 1, outside_temperature=72, debug=True))
    print(mpc_thermal_model.predict(70, 0, outside_temperature=72, debug=True))
    print(mpc_thermal_model.predict(70, 2, outside_temperature=72, debug=True))
