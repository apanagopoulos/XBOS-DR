import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import utils
from ParentThermalModel import ParentThermalModel

import yaml
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


# following model also works as a sklearn model.
class ThermalModel(ParentThermalModel):
    def __init__(self, thermal_precision=0.05, learning_rate=0.00001):
        '''
        
        :param thermal_precision: the closest multiple of which to round to.
        '''

        self._params = None
        self._params_coeff_order = None # first part of _params is coeff part
        self._params_bias_order = None # the rest is bias part.
        self._filter_columns = None  # order of columns by which to filter when predicting and fitting data.

        self.thermal_precision = thermal_precision
        self.learning_rate = learning_rate  # TODO evaluate which one is best.
        super(ThermalModel, self).__init__(thermal_precision)

    # thermal model function
    def _func(self, X, *params):
        """The polynomial with which we model the thermal model.
        :param X: np.array with row order (Tin, action, Tout, dt, rest of zone temperatures). Is also compatible 
                    with pd.df with columns ('t_in', 'action', 't_out', 'dt', "zone_temperatures")
        :param *coeff: the coefficients for the thermal model. 
                Should be in order: self._prams_coeff_order, self._params_bias_order
        """
        # Check if we have the right data type.
        if isinstance(X, pd.DataFrame):
            X = X[self._filter_columns].T.as_matrix()
        elif not isinstance(X, np.ndarray):
            raise Exception("_func did not receive a valid datatype. Expects pd.df or np.ndarray")

        if not params:
            try:
                getattr(self, "_params")
            except AttributeError:
                raise RuntimeError("You must train classifer before predicting data!")
            params = self._params

        coeffs = params[:len(self._params_coeff_order)]
        biases = params[len(self._params_coeff_order):]

        features = self._features(X)
        Tin, action = X[0], X[1]

        action_filter = self._filter_actions(X)
        features_biases = (features - biases) * action_filter

        # print("fil",action_filter)
        # print("coeffs", coeffs)
        # print("bias", biases)
        # print("featbias", features_biases)
        # print(X.T)
        return Tin + features_biases.dot(np.array(coeffs))


    def _features(self, X):
        """Returns the features we are using as a matrix.
        :param X: A matrix with row order (Tin, action, Tout, dt, rest of zone temperatures)
        :return np.matrix. each column corresponding to the features in the order of self._param_order"""
        Tin, action, Tout, dt, zone_temperatures = X[0], X[1], X[2], X[3], X[4:]
        features = [Tin,  # action == utils.HEATING_ACTION
                    Tin,  # action == utils.COOLING_ACTION
                    Tin,  # action == utils.TWO_STAGE_HEATING_ACTION
                    Tin,  # action == utils.TWO_STAGE_COOLING_ACTION
                    Tin - Tout,
                    np.zeros(X.shape[1])]  # overall bias
        for zone_temp in zone_temperatures:
            features.append(Tin - zone_temp)
        return np.array(features).T

    def _filter_actions(self, X):
        """Returns a matrix of _features(X) shape which tells us which features to use. For example, if we have action Heating,
        we don't want to learn cooling coefficients, so we set the cooling feature to zero.
        :param X: A matrix with row order (Tin, action, Tout, dt, rest of zone temperatures)
        :return np.matrix. each column corresponding to whether to use the features in the order of self._param_order"""
        num_data = X.shape[1]
        action, zone_temperatures = X[1], X[4:]
        action_filter = [action == utils.HEATING_ACTION,
                  action == utils.COOLING_ACTION,
                  action == utils.TWO_STAGE_HEATING_ACTION,
                  action == utils.TWO_STAGE_COOLING_ACTION,
                  np.ones(num_data),  # tout
                  np.ones(num_data)]  # bias

        for _ in zone_temperatures:
            action_filter.append(np.ones(num_data))

        action_filter = np.array(action_filter).T
        return action_filter

    def fit(self, X, y):
        # TODO how should it update parameters when given more new data?
        """Needs to be called to initally fit the model. Will set self._params to coefficients.
        Will refit the model if called with new data.
        :param X: pd.df with columns ('t_in', 'action', 't_out', 'dt') and all zone temperature where all have
        to begin with "zone_temperature_" + "zone name"
        :param y: the labels corresponding to the data. As a pd.dataframe
        :return self
        """
        zone_col = X.columns[["zone_temperature_" in col for col in X.columns]]
        filter_columns = ['t_in', 'action', 't_out', 'dt'] + list(zone_col)

        # give mapping from params to coefficients and to store the order in which we get the columns.
        self._filter_columns = filter_columns
        self._params_coeff_order = ["heating", 'cooling',
                                    'two_stage_heating', 'two_stage_cooling',
                                    't_out', 'bias'] + list(zone_col)

        self._params_bias_order = ["heating", 'cooling',
                                    'two_stage_heating', 'two_stage_cooling',
                                    't_out', 'bias'] + list(zone_col)

        # fit the data. we start our guess with all ones for coefficients.
        # Need to do so to be able to generalize to variable number of zones.
        popt, pcov = curve_fit(self._func, X[filter_columns].T.as_matrix(), y.as_matrix(),
                               p0=np.ones(len(
                                   self._params_coeff_order) + len(self._params_bias_order)))
        self._params = np.array(popt)
        return self

    def update_fit(self, X, y):
        # does not fit to the current function anymore.
        return
        """Adaptive Learning for one datapoint. The data given will all be given the same weight when learning.
        :param X: (pd.df) with columns ('t_in', 'action', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name
        :param y: (float)"""
        # NOTE: Using gradient decent $$self.params = self.param - self.learning_rate * 2 * (self._func(X, *params) - y) * features(X)$$
        loss = self._func(X[self._filter_columns].T.as_matrix(), *self._params)[0] - y
        adjust = self.learning_rate * loss * self._features(X[self._filter_columns].T.as_matrix())
        self._params = self._params - adjust.reshape(
            (adjust.shape[0]))  # to make it the same dimensions as self._params




class MPCThermalModel(ThermalModel):
    """Class specifically designed for the MPC process. A child class of ThermalModel with functions
        designed to simplify usage."""

    def __init__(self, zone, thermal_data, interval_length, thermal_precision=0.05):
        """
        :param zone: The zone this Thermal model is meant for. 
        :param thermal_data: pd.df thermal data for zone (as preprocessed by ControllerDataManager). Only used for fitting.
        :param interval_length: (int) Number of minutes between
        :param thermal_precision: (float) The increment to which to round predictions to. (e.g. 1.77 becomes 1.75
         and 4.124 becomes 4.10)
        """
        self.zone = zone
        thermal_data = thermal_data.rename({"temperature_zone_" + self.zone: "t_in"}, axis="columns")

        # set our parent up first
        super(MPCThermalModel, self).__init__(thermal_precision=thermal_precision) # TODO What to do with Learning rate ?
        super(MPCThermalModel, self).fit(thermal_data, thermal_data["t_next"])

        self._oldParams = {}

        self.interval = interval_length  # new for predictions. Will be fixed right?

        self.zoneTemperatures = None
        self.weatherPredictions = None  # store weather predictions for whole class

        self.lastAction = None  # TODO Fix, absolute hack and not good. controller should store this.

    # TODO Fix, absolute hack and not good. controller should store this.
    def set_last_action(self, action):
        self.lastAction = action

    def set_weather_predictions(self, weatherPredictions):
        self.weatherPredictions = weatherPredictions

    def _datapoint_to_dataframe(self, interval, action, t_out, zone_temperatures):
        """A helper function that converts a datapoint to a pd.df used for predictions.
        Assumes that we have self.zone"""
        X = {"dt": interval, "action": action,
             "t_out": t_out}
        for key_zone, val in zone_temperatures.items():
            if key_zone != self.zone:
                X["zone_temperature_" + key_zone] = val
            else:
                X["t_in"] = val

        return pd.DataFrame(X, index=[0])

    def set_temperatures_and_fit(self, curr_zone_temperatures, interval, now):
        """
        performs one update step for the thermal model and
        stores curr temperature for every zone. Call whenever we are starting new interval.
        :param curr_zone_temperatures: {zone: temperature}
        :param interval: The delta time since the last action was called. 
        :param now: the current time in the timezone as weather_predictions.
        :return: None
        """
        # TODO Fix this to get online learning going.
        return
        # store old temperatures for potential fitting
        old_zone_temperatures = self.zoneTemperatures

        # set new zone temperatures.
        self.zoneTemperatures = curr_zone_temperatures

        # TODO can't fit? should we allow?
        if self.lastAction is None or self.weatherPredictions is None:
            return

        action = self.lastAction # TODO get as argument?

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

    def predict(self, t_in, zone, action, time=-1, outside_temperature=None, interval=None):
        """
        Predicts temperature for zone given.
        :param t_in: 
        :param zone: 
        :param action: (float)
        :param outside_temperature: 
        :param interval: 
        :param time: the hour index for self.weather_predictions. 
        TODO understand how we can use hours if we look at next days .(i.e. horizon extends over midnight.)
        :return: (array) predictions in order
        """
        if interval is None:
            interval = self.interval
        if outside_temperature is None:
            assert time != -1
            assert self.weatherPredictions is not None
            t_out = self.weatherPredictions[time]
        else:
            t_out = outside_temperature

        # TODO NEED TO FIX THIS
        zone_temps = self.zoneTemperatures
        zone_temps[self.zone] = t_in


        X = self._datapoint_to_dataframe(interval, action, t_out, zone_temps) # TODO which t_in are we really assuming?
        return super(MPCThermalModel, self).predict(X)

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


if __name__ == '__main__':
    print("Thermal Model")
    import pickle
    import sys


    sys.path.append("./ThermalModels")
    from AverageThermalModel import AverageThermalModel
    building = "avenal-animal-shelter"
    thermal_data = utils.get_data(building=building, days_back=100, evaluate_preprocess=True, force_reload=False)

    model = ThermalModel()
    avg_model = AverageThermalModel()
    zone, zone_data = thermal_data.items()[0]

    zone_data = zone_data[zone_data["dt"] == 5]
    zone_data = zone_data[zone_data["t_min"] != zone_data["t_max"]]

    zone_data = zone_data[((zone_data["t_in"] > zone_data["t_next"]) | (zone_data["action"] != utils.COOLING_ACTION))]

    cooling_data = zone_data[(zone_data["t_in"] > zone_data["t_next"]) & (zone_data["action"] == utils.COOLING_ACTION)]

    with open("weird", "wb") as f:
        pickle.dump(zone_data, f)

    print(zone)
    model.fit(zone_data, zone_data["t_next"])
    avg_model.fit(zone_data, zone_data["t_next"])

    for i in range(-1, 6):
        print("normal", model.score(cooling_data, cooling_data["t_next"], scoreType=i))
        print("avg", avg_model.score(cooling_data, cooling_data["t_next"], scoreType=i))

    print("coeff", model._params[:len(model._params_coeff_order)])
    print("bias", model._params[len(model._params_coeff_order):])

    print(model._params_coeff_order)
    print(model._params_bias_order)

    print("avg coeff", avg_model._params)

    utils.apply_consistency_check_to_model(thermal_model=model)

    #(Tin, action, Tout, dt, rest of zone temperatures)
    X = np.array([[75, 2, 75, 5, 75, 75, 75]]).T
    # print(model._func(X))
    # print(model.predict(X))


    # r = therm_data["HVAC_Zone_Shelter_Corridor"].iloc[-1]
    # print(r)
    # print model.predict(t_in=r["t_in"], zone="HVAC_Zone_Shelter_Corridor", action=r["action"],outside_temperature=r["t_out"], interval=r["dt"])
    # print("hi")
