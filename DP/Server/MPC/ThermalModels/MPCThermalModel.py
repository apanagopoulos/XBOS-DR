import sys

import numpy as np
import pandas as pd
import yaml

# daniel imports
sys.path.append("./ThermalModels")
sys.path.append("..")

import utils
from DP.Server.MPC.ThermalModels.ThermalModel import ThermalModel
from AverageThermalModel import AverageThermalModel
from ConstantThermalModel import ConstantThermalModel


class MPCThermalModel:
    """Class specifically designed for the MPC process. A child class of ThermalModel with functions
        designed to simplify usage.
        
        NOTE: when predicting will always round."""

    def __init__(self, zone, thermal_data, interval_length, thermal_precision=0.05, is_two_stage=False):
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
        self.thermal_model = ThermalModel(thermal_precision=thermal_precision)
        self.average_thermal_model = AverageThermalModel(thermal_precision=thermal_precision)
        self.constant_thermal_model = ConstantThermalModel(thermal_precision=thermal_precision)

        # the constants to use for the constant thermal model. It will add this constant to t_in to get t_next.
        # Making sure we are adding increments for thermal precision to make use of pruning in the DP graph.
        # TODO Get variable constants.
        # Using thermal precision to predict if both average and thermal model provide
        # inconsistent or non sensible data.
        no_action_constant = 0
        heating_constant = thermal_precision
        cooling_constant = -thermal_precision

        # fit the thermal models
        self.thermal_model = self.thermal_model.fit(thermal_data, thermal_data["t_next"])
        self.average_thermal_model = self.average_thermal_model.fit(thermal_data, thermal_data["t_next"])
        self.constant_thermal_model = self.constant_thermal_model.fit(thermal_data, thermal_data["t_next"],
                                                                      [no_action_constant,
                                                                       heating_constant,
                                                                       cooling_constant])  # constant action params.

        # online training debugging.
        self._oldParams = {}

        # in case we want to set it during creation and don't have access during prediction.
        self.interval = interval_length

        # the data that will be stored through the MPCThermal Model to make usage easier.
        self.zoneTemperatures = None
        self.weatherPredictions = None  # for now, the hour from now corresponds to index nr. We always start in 0th hour.
        self.lastAction = None

    def set_last_action(self, action):
        """Set the action that was last used by MPC for later online training.
        :param action: (int) action nr as given in utils."""
        self.lastAction = action

    def set_weather_predictions(self, weather_predictions):
        """Set weather predictions after last fit to store for later predictions.
        :param weather_predictions: (float []) 0th index corresponds to t_out now and 1st index to the temperature 
        outside in one hour from now, etc. Now is defined as the start time of the advise graph."""
        self.weatherPredictions = weather_predictions

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

        return pd.DataFrame(X, index=[0])

    def set_temperatures_and_fit(self, curr_zone_temperatures, interval=15):
        """
        performs one update step for the thermal model and
        stores curr temperature for every zone. Call whenever we are starting new interval.
        :param curr_zone_temperatures: {zone: temperature}
        :param interval: The delta time since the last action was called. 
        :return: None
        """

        # store old temperatures for potential fitting
        old_zone_temperatures = self.zoneTemperatures

        # set new zone temperatures.
        self.zoneTemperatures = curr_zone_temperatures

        # TODO Fix this to get online learning going.
        return

        # TODO can't fit? should we allow?
        if self.lastAction is None or self.weatherPredictions is None:
            return

        action = self.lastAction  # TODO get as argument?

        t_out = self.weatherPredictions[now.hour]

        y = curr_zone_temperatures[self.zone]
        X = self._datapoint_to_dataframe(interval, action, t_out, old_zone_temperatures)

        # make sure we have no nan values. TODO make ure we are checking better.
        assert not X.isnull().values.any()

        # online learning the new data
        super(MPCThermalModel, self).updateFit(X, y)

        # TODO DEBUG MODE?
        # # store the params for potential evaluations.
        # self._oldParams[zone].append(self.zoneThermalModels[zone]._params)
        # # to make sure oldParams holds no more than 50 values for each zone
        # self._oldParams[zone] = self._oldParams[zone][-50:]

    def _prediction_test(self, X, thermal_model):
        """Takes the data X and for each datapoint runs the thermal model for each possible actions and 
        sees if the prediction we would get is consistent and sensible. i.e. temperature change for heating is more than for 
        cooling etc. 
        Note, relies on having set self.is_two_stage to know if we are predicting two stage cooling data. """

        # get predictions for every action possible.
        def predict_action(X, action, thermal_model):
            unit_actions = np.ones(X.shape[0])
            X_copy = X.copy()
            X_copy["action"] = unit_actions * action
            return thermal_model.predict(X_copy)

        def consistency_check(row, is_two_stage):
            """check that the temperatures are in the order we expect them. Heating has to be strictly more than no
            action and cooling and cooling has to be strictly less than heating and no action.
            If we only want to check consistency for the doing nothing action, then it should suffice to only check
            that the max of cooling is smaller and the min of heating is larger. 
            :param row: pd.df keys which are the actions in all caps and "MAIN_ACTION" ,the action we actually 
                    want to find the consistency for.
            :param is_two_stage: bool wether we are using predictions for a two stage building.
            :return bool. Whether the given data is consistent."""
            main_action = row["MAIN_ACTION"]

            if is_two_stage:
                if main_action == utils.NO_ACTION:
                    if max(row["COOLING_ACTION"], row["TWO_STAGE_COOLING_ACTION"]) \
                            < row["NO_ACTION"] \
                            < min(row["HEATING_ACTION"], row["TWO_STAGE_HEATING_ACTION"]):
                        return True
                else:
                    # TODO Maybe have only strictly greater.
                    if row["TWO_STAGE_COOLING_ACTION"] <= row["COOLING_ACTION"] \
                            < row["NO_ACTION"] \
                            < row["HEATING_ACTION"] <= row["TWO_STAGE_HEATING_ACTION"]:
                        return True
            else:
                if row["COOLING_ACTION"] < row["NO_ACTION"] < row["HEATING_ACTION"]:
                    return True

            return False

        def sensibility_check(X, is_two_stage, sensibility_measure=20):
            """Checks if the predictions are within sensibility_measure degrees of tin. It wouldn't make sense to predict 
            more. This will be done for all possible action predictions. If any one is not sensible, we will disregard 
            the whole prediction set. 
            ALSO, check if Nan values
            :param X: pd.df keys which are the actions in all caps and "MAIN_ACTION" ,the action we actually 
                    want to find the consistency for.
            :param is_two_stage: bool wether we are using predictions for a two stage building.
            :param sensibility_measure: (Float) degrees within the prediction may lie. 
                            e.g. t_in-sensibility_measure < prediction < t_in+sensibility_measure
            :return np.array booleans"""
            differences = []
            differences.append(np.abs(X["NO_ACTION"] - X["T_IN"]))
            differences.append(np.abs(X["HEATING_ACTION"] - X["T_IN"]))
            differences.append(np.abs(X["COOLING_ACTION"] - X["T_IN"]))
            if is_two_stage:
                differences.append(np.abs(X["TWO_STAGE_HEATING_ACTION"] - X["T_IN"]))
                differences.append(np.abs(X["TWO_STAGE_COOLING_ACTION"] - X["T_IN"]))

            # check if every difference is in sensible band and not nan. We can check if the prediction is nan
            # by checking if the difference is nan, because np.nan + x = np.nan
            sensibility_filter_array = [(diff < sensibility_measure) & (diff != np.nan) for diff in differences]
            # putting all filters together by taking the and of all of them.
            sensibility_filter_check = reduce(lambda x, y: x & y, sensibility_filter_array)
            return sensibility_filter_check.values


        no_action_predictions = predict_action(X, utils.NO_ACTION, thermal_model)
        heating_action_predictions = predict_action(X, utils.HEATING_ACTION, thermal_model)
        cooling_action_predictions = predict_action(X, utils.COOLING_ACTION, thermal_model)
        if self.is_two_stage:
            two_stage_heating_predictions = predict_action(X, utils.TWO_STAGE_HEATING_ACTION, thermal_model)
            two_stage_cooling_predictions = predict_action(X, utils.TWO_STAGE_COOLING_ACTION, thermal_model)
        else:
            two_stage_cooling_predictions = None
            two_stage_heating_predictions = None

        predictions_action = pd.DataFrame({"T_IN": X["t_in"], "NO_ACTION": no_action_predictions,
                                           "HEATING_ACTION": heating_action_predictions,
                                           "COOLING_ACTION": cooling_action_predictions,
                                           "TWO_STAGE_HEATING_ACTION": two_stage_heating_predictions,
                                           "TWO_STAGE_COOLING_ACTION": two_stage_cooling_predictions,
                                           "MAIN_ACTION": X["action"]})

        consistent_filter = predictions_action.apply(lambda row: consistency_check(row, self.is_two_stage), axis=1)
        sensibility_filter = sensibility_check(predictions_action, self.is_two_stage, sensibility_measure=20)

        return consistent_filter.values & sensibility_filter

    # TODO right now only one datapoint is being predicted. Extend to more. Consistecy check works for arbitrary data.
    def predict(self, t_in, action, time=None, outside_temperature=None, interval=None, debug=False):
        """
        Predicts temperature for zone given.
        :param t_in: 
        :param action: (float)
        :param outside_temperature: 
        :param interval: 
        :param time: the hour index for self.weather_predictions. 
        TODO understand how we can use hours if we look at next days .(i.e. horizon extends over midnight.)
        :param debug: wether to debug, meaning return the type of thermal model that causes each prediction.
        :return: not debug: (np.array) predictions in order. 
                 debug: (np.array) predictions in order, (np.array dtype=object/strings) thermal_model types 
        """
        if interval is None:
            interval = self.interval
        if outside_temperature is None:
            assert time is not None
            assert self.weatherPredictions is not None
            t_out = self.weatherPredictions[time]
        else:
            t_out = outside_temperature

        X = self._datapoint_to_dataframe(interval, t_in, action, t_out, self.zoneTemperatures)

        thermal_model_predictions = self.thermal_model.predict(X)

        # makes sure that the predictions that we make are consistent.
        thermal_model_filter = self._prediction_test(X, self.thermal_model)

        # identify all the elements that are predicted by thermal_model
        model_types = np.array(thermal_model_filter.shape, dtype=object)
        model_types[thermal_model_filter] = "thermal_model"

        # if not all thermal_model predictions are sensible and consistent
        if not all(thermal_model_filter):
            thermal_model_inconsistent_data = X[thermal_model_filter == 0]
            average_thermal_model_predictions = self.average_thermal_model.predict(thermal_model_inconsistent_data)
            average_thermal_model_filter = self._prediction_test(thermal_model_inconsistent_data,
                                                                 self.average_thermal_model)

            # identify all the elements that are predicted by average thermal_model
            average_model_types = np.array(average_thermal_model_filter.shape, dtype=object)
            average_model_types[average_thermal_model_filter] = "average_thermal_model"

            # if not all average_thermal_model predictions are sensible and consistent
            if not all(average_thermal_model_filter):
                average_model_inconsistent_data = thermal_model_inconsistent_data[
                    average_thermal_model_filter == 0]
                constant_thermal_model_predictions = self.constant_thermal_model.predict(
                    average_model_inconsistent_data)
                constant_thermal_model_filter = self._prediction_test(
                    average_model_inconsistent_data, self.constant_thermal_model)

                # Sanity check that the constant model gives sensible and consistent data.
                assert all(constant_thermal_model_filter)

                # make average consistent by filling it with constant predictions.
                average_thermal_model_predictions[
                    average_thermal_model_filter == 0] = constant_thermal_model_predictions

                # identify all the elements that are predicted by constant thermal_model
                average_model_types[average_thermal_model_filter == 0] = "constant_model"

            # make thermal model consistent by using the corrected average thermal model predictions
            thermal_model_predictions[thermal_model_filter == 0] = average_thermal_model_predictions

            # Combine all prediction types
            model_types[thermal_model_filter == 0] = average_model_types

        if debug:
            return thermal_model_predictions, model_types

        return thermal_model_predictions

    def save_to_config(self):
        # this does not work anymore as intended.
        return

        """saves the whole model to a yaml file.
        RECOMMENDED: PYAML should be installed for prettier config file."""
        config_dict = {}

        # store zone temperatures
        config_dict["Zone Temperatures"] = self.zoneTemperatures

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
