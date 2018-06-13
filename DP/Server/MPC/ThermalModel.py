import numpy as np
import pandas as pd

import yaml
from scipy.optimize import curve_fit
# daniel imports
from sklearn.base import BaseEstimator, RegressorMixin


# following model also works as a sklearn model.
class ThermalModel(BaseEstimator, RegressorMixin):
    def __init__(self, thermal_precision=0.05, learning_rate=0.00001, scoreType=-1, ):
        '''
        _params:
            scoreType: (int) which actions to filter by when scoring. -1 indicates no filter, 0 no action,
                        1 heating, 2 cooling.
        :param thermal_precision: the number of decimal points when predicting.
        '''
        self.scoreType = scoreType  # instance variable because of how cross validation works with sklearn

        self._params = None
        self._params_order = None
        self._filter_columns = None  # order of columns by which to filter when predicting and fitting data.

        # NOTE: if wanting to use cross validation, put these as class variables.
        #  Also, change in score, e.g. self.model_error to ThermalModel.model_error
        # keeping track of all the rmse's computed with this class.
        # first four values are always the training data errors.
        self.baseline_error = []
        self.model_error = []
        self.scoreTypeList = []  # to know which action each rmse belongs to.
        self.betterThanBaseline = []

        self.thermalPrecision = thermal_precision
        self.learning_rate = learning_rate  # TODO evaluate which one is best.

    # thermal model function
    def _func(self, X, *coeff):
        """The polynomial with which we model the thermal model.
        :param X: np.array with column order (Tin, a1, a2, Tout, dt, rest of zone temperatures)
        :param *coeff: the coefficients for the thermal model. Should be in order: a1, a2, (Tout - Tin),
         bias, zones coeffs (as given by self._params_order)
        """
        features = self._features(X)
        Tin = X[0]
        return Tin + features.T.dot(np.array(coeff))

    def _features(self, X):
        """Returns the features we are using as a matrix.
        :param X: A matrix with column order (Tin, a1, a2, Tout, dt, rest of zone temperatures)
        :return np.matrix. each column corresponding to the features in the order of self._param_order"""
        Tin, a1, a2, Tout, dt, zone_temperatures = X[0], X[1], X[2], X[3], X[4], X[5:]
        features = [a1, a2, Tout - Tin, np.ones(X.shape[1])]
        for zone_temp in zone_temperatures:
            features.append(zone_temp - Tin)

        return np.array(features) * dt

    def fit(self, X, y=None):
        """Needs to be called to fit the model. Will set self._params to coefficients. 
        :param X: pd.df with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name"
        :param y: the labels corresponding to the data. As a pd.dataframe
        :return self
        """
        zone_col = X.columns[["zone_temperature_" in col for col in X.columns]]
        filter_columns = ['t_in', 'a1', 'a2', 't_out', 'dt'] + list(zone_col)

        # give mapping from params to coefficients and to store the order in which we get the columns.
        self._filter_columns = filter_columns
        self._params_order = ["a1", 'a2', 't_out', 'bias'] + list(zone_col)

        # fit the data. we start our guess with all ones for coefficients.
        # Need to do so to be able to generalize to variable number of zones.
        popt, pcov = curve_fit(self._func, X[filter_columns].T.as_matrix(), y.as_matrix(),
                               p0=np.ones(len(
                                   self._params_order)))
        self._params = np.array(popt)
        return self

    def updateFit(self, X, y):
        """Adaptive Learning for one datapoint. The data given will all be given the same weight when learning.
        :param X: (pd.df) with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name
        :param y: (float)"""
        # NOTE: Using gradient decent $$self.params = self.param - self.learning_rate * 2 * (self._func(X, *params) - y) * features(X)
        loss = self._func(X[self._filter_columns].T.as_matrix(), *self._params)[0] - y
        adjust = self.learning_rate * loss * self._features(X[self._filter_columns].T.as_matrix())
        self._params = self._params - adjust.reshape(
            (adjust.shape[0]))  # to make it the same dimensions as self._params

    def predict(self, X, y=None, should_round=True):
        """Predicts the temperatures for each row in X.
        :param X: pd.df/pd.Series with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperatures where all 
        have to begin with "zone_temperature_" + "zone name"
        :return (np.array) entry corresponding to prediction of row in X.
        """
        # only predicts next temperatures
        try:
            getattr(self, "_params")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        # assumes that pandas returns df in order of indexing when doing X[self._filter_columns].
        predictions = self._func(X[self._filter_columns].T.as_matrix(), *self._params)
        if should_round:
            # source for rounding: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
            return self.thermalPrecision * np.round(predictions / float(self.thermalPrecision))
        else:
            return predictions

    def _normalizedRMSE_STD(self, prediction, y, dt):
        '''Computes the RMSE with scaled differences to normalize to 15 min intervals.
        NOTE: Use method if you already have predictions.'''
        diff = prediction - y

        # to offset for actions which were less than 15 min. Normalizes to 15 min intervals.
        # TODO maybe make variable standard intervals.
        diff_scaled = diff * 15. / dt
        mean_error = np.mean(diff_scaled)
        rmse = np.sqrt(np.mean(np.square(diff_scaled)))
        # standard deviation of the error
        diff_std = np.sqrt(np.mean(np.square(diff_scaled - mean_error)))
        return mean_error, rmse, diff_std

    def score(self, X, y, scoreType=None):
        """Scores the model on the dataset given by X and y."""
        if scoreType is None:
            scoreType = self.scoreType
        assert scoreType in list(range(-1, 4))

        self.scoreTypeList.append(scoreType)  # filter by the action we want to score by
        if scoreType == 0:
            filter_arr = (X['a1'] == 0) & (X['a2'] == 0)
        elif scoreType == 1:
            filter_arr = X['a1'] == 1
        elif scoreType == 2:
            filter_arr = X['a2'] == 1
        else:
            filter_arr = np.ones(X['a1'].shape) == 1

        X = X[filter_arr]
        y = y[filter_arr]

        prediction = self.predict(X)  # only need to predict for relevant actions

        mean_error, rmse, std = self._normalizedRMSE_STD(prediction, y, X['dt'])

        # add model RMSE for reference.
        self.model_error.append({"mean": mean_error, "rmse": rmse, "std": std})

        # add trivial error for reference.
        trivial_mean_error, trivial_rmse, trivial_std = self._normalizedRMSE_STD(X['t_in'], y, X['dt'])
        self.baseline_error.append({"mean": trivial_mean_error, "rmse": trivial_rmse, "std": trivial_std})

        # to keep track of whether we are better than the baseline/trivial
        self.betterThanBaseline.append(trivial_rmse > rmse)

        return rmse


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
        Assumes that we have self.zoneTemperatures and self.zone"""
        X = {"dt": interval, "a1": int(0 < action <= 1), "a2": int(1 < action <= 2),
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
        :param zone_temps: {zone: temperature}
        :param interval: The delta time since the last action was called. 
        :param now: the current time in the timezone as weather_predictions.
        :return: None
        """
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
    with open("../Buildings/ciee/ZoneConfigs/HVAC_Zone_CentralZone.yml", 'r') as ymlfile:
        advise_cfg = yaml.load(ymlfile)

    import pickle

    # therm_data_file = open("../Thermal Data/ciee_thermal_data_demo")
    # therm_data = pickle.load(therm_data_file)
    #
    # therm_data_file.close()
    #
    # mpcThermalModel = MPCThermalModel(therm_data, 15)
    #
    # with open("../Thermal Data/thermal_model_demo", "wb") as f:
    #     pickle.dump(mpcThermalModel, f)
    with open("../Thermal Data/ciee_thermal_data_demo") as f:
        therm_data = pickle.load(f)

    model = MPCThermalModel("HVAC_Zone_Southzone", therm_data["HVAC_Zone_Southzone"], 15)

    # r = therm_data["HVAC_Zone_Shelter_Corridor"].iloc[-1]
    # print(r)
    # print model.predict(t_in=r["t_in"], zone="HVAC_Zone_Shelter_Corridor", action=r["action"],outside_temperature=r["t_out"], interval=r["dt"])
    # print("hi")
