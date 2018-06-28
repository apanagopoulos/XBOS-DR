
import numpy as np
import pandas as pd

import yaml
from scipy.optimize import curve_fit
# daniel imports
from sklearn.base import BaseEstimator, RegressorMixin

# used to find best thermal model.
class ThermalModel(BaseEstimator, RegressorMixin):
    def __init__(self, thermal_precision=0.05, learning_rate=0.00001):
        '''

        :param thermal_precision: the number of decimal points when predicting.
        '''

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

        self.thermal_precision = thermal_precision
        self.learning_rate = learning_rate  # TODO evaluate which one is best.

    # ========== average learning ========
    # just learning the average of each action and predict with that.
    # thermal model function
    def _func(self, X, *coeff):
        """The polynomial with which we model the thermal model.
        :param X: np.array with row order (Tin, a1, a2, Tout, dt, rest of zone temperatures)
        :param *coeff: the averages for the thermal model. Should have no_action, heating, cooling as order.
        """
        X = X.T # making each data point a row
        a0_delta, a1_delta, a2_delta = coeff
        # action idx
        Tin = 0
        a1 = 1
        a2 = 2
        dt = 4

        t_next = []
        for i in range(X.shape[0]):
            data_point = X[i]
            if data_point[a1] == 1:
                t_next.append(data_point[Tin] + a1_delta ) # * data_point[dt])
            elif data_point[a2] == 1:
                t_next.append(data_point[Tin] + a2_delta ) # * data_point[dt])
            else:
                t_next.append(data_point[Tin] + a0_delta ) # * data_point[dt])

        return np.array(t_next)

    # Fit without zone interlocking.
    def fit(self, X, y):
        # TODO how should it update parameters when given more new data?
        """Needs to be called to initally fit the model. Will set self._params to coefficients.
        Will refit the model if called with new data.
        :param X: pd.df with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have
        to begin with "zone_temperature_" + "zone name"
        :param y: the labels corresponding to the data. As a pd.dataframe
        :return self
        """

        filter_columns = ['t_in', 'a1', 'a2']
        # give mapping from params to coefficients and to store the order in which we get the columns.
        self._filter_columns = filter_columns


        def get_delta_mean(action_data):
            # get the mean change of temperature from now to next. Normalized
            # TODO assuming action_data has "t_next"
            # TODO assuming that mean will be a float
            return np.mean(y - action_data["t_in"]) # /action_data["dt"])

        cooling_data = X[X["a2"] == 1]
        heating_data = X[X["a1"] == 1]
        no_action_data = X[(X["a1"] == 0) & (X["a2"] == 0)]

        mean_cooling_delta = get_delta_mean(cooling_data)
        mean_heating_delta = get_delta_mean(heating_data)
        mean_no_action_delta = get_delta_mean(no_action_data)
        self._params = np.array([mean_no_action_delta, mean_heating_delta, mean_cooling_delta])

        def average_quality_check(*params):
            no_action_average, heating_average, cooling_average = params

            consistency_flag = True

            if heating_average <= 0:
                consistency_flag = False
                print("Warning, heating_average is lower than 0.")
            if cooling_average >= 0:
                consistency_flag = False
                print("Warning, cooling_average is higher than 0.")

            # check that heating is more than no action and cooling
            if heating_average <= no_action_average or heating_average <= cooling_average:
                consistency_flag = False
                print("Warning, heating_average is too low compared to other actions.")
            # check cooling is lower than heating and no action
            if cooling_average >= no_action_average or cooling_average >= heating_average:
                consistency_flag = False
                print("Warning, cooling_average is too high compared to other actions.")
            # check if no action is between cooling and heating
            if not cooling_average < no_action_average < heating_average:
                consistency_flag = False
                print("Warning, no_action_average is not inbetween heating temperature and cooling temperature.")

            # want to know for what data it didn't work
            if not consistency_flag:
                print("Inconsistency for following parameters:")
                print("No Action: %f " % no_action_average)
                print("Heating: %f " % heating_average)
                print("Cooling: %f " % cooling_average)

            return consistency_flag
        assert average_quality_check(*self._params)

        return self


    def updateFit(self, X, y):
        """Adaptive Learning for one datapoint. The data given will all be given the same weight when learning.
        :param X: (pd.df) with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name
        :param y: (float)"""
        pass

    # ==== average model ends. ======

    def predict(self, X, should_round=False):
        # TODO CHange in Advise class. Say it should round.
        """Predicts the temperatures for each row in X.
        :param X: pd.df with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperatures where all 
        have to begin with "zone_temperature_" + "zone name"
        :param should_round: bool. Wether to round the prediction according to self.thermal_precision.
        :return (np.array) entry corresponding to prediction of row in X.
        """
        # only predicts next temperatures
        try:
            getattr(self, "_params")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        # assumes that pandas returns df in order of indexing when doing X[self._filter_columns].
        predictions = self._func(X[self._filter_columns].T.values, *self._params)
        if should_round:
            # source for rounding: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
            return self.thermal_precision * np.round(predictions / float(self.thermal_precision))
        else:
            return predictions

    def _normalizedRMSE_STD(self, prediction, y, dt):
        '''Computes the RMSE with scaled differences to normalize to 15 min intervals.
        NOTE: Use method if you already have predictions.
        :param prediction: (np.array) list of predictions.
        :param y: (np.array) list of expected values.
        :param dt: (np.array) list of dt for each prediction. (i.e. time between t_in and t_next)
        :return: mean_error (mean of prediction - y), rmse, std (std of the error). All data properly normalized to 15 min
        intervals.'''
        diff = prediction - y

        # to offset for actions which were less than 15 min. Normalizes to 15 min intervals.
        # TODO maybe make variable standard intervals.
        # TODO Maybe no scales.
        diff_scaled = diff # * 15. / dt
        mean_error = np.mean(diff_scaled)
        # root mean square error
        rmse = np.sqrt(np.mean(np.square(diff_scaled)))
        # standard deviation of the error
        diff_std = np.sqrt(np.mean(np.square(diff_scaled - mean_error)))
        return mean_error, rmse, diff_std

    def score(self, X, y, scoreType=-1):
        """Scores the model on the dataset given by X and y.
                :param X: (pd.df) with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have 
                to begin with "zone_temperature_" + "zone name
                :param y: (float array)"""

        # Filters data to score only on subset of actions.
        assert scoreType in list(range(-1, 4))
        # filter by the action we want to score by
        if scoreType == 0:
            filter_arr = (X['a1'] == 0) & (X['a2'] == 0)
        elif scoreType == 1:
            filter_arr = X['a1'] == 1
        elif scoreType == 2:
            filter_arr = X['a2'] == 1
        elif scoreType == -1:
            filter_arr = np.ones(X.shape[0]) == 1

        X = X[filter_arr]
        y = y[filter_arr]

        # Predict on filtered data
        prediction = self.predict(X)  # only need to predict for relevant actions. No rounding when evaluating.

        # Get model error
        mean_error, rmse, std = self._normalizedRMSE_STD(prediction, y, X['dt'])
        self.model_error.append({"mean": mean_error, "rmse": rmse, "std": std})

        # add trivial error for reference. Trivial error assumes that the temperature in X["dt"] time will be the same
        # as the one at the start of interval.
        trivial_mean_error, trivial_rmse, trivial_std = self._normalizedRMSE_STD(X['t_in'], y, X['dt'])
        self.baseline_error.append({"mean": trivial_mean_error, "rmse": trivial_rmse, "std": trivial_std})

        # to keep track of whether we are better than the baseline/trivial
        self.betterThanBaseline.append(trivial_rmse > rmse)

        # To know which actions we scored on
        self.scoreTypeList.append(scoreType)

        return rmse

class AverageMPCThermalModel(ThermalModel):
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
        super(AverageMPCThermalModel, self).__init__(thermal_precision=thermal_precision) # TODO What to do with Learning rate ?
        super(AverageMPCThermalModel, self).fit(thermal_data, thermal_data["t_next"])

        self._oldParams = {}

        self.interval = interval_length  # new for predictions. Will be fixed right?

        self.zoneTemperatures = None
        self.weatherPredictions = None  # store weather predictions for whole class

        self.lastAction = None  # TODO Fix, absolute hack and not good. controller should store this.

    # TODO Fix, absolute hack and not good. controller should store this.
    def set_last_action(self, action):
        pass

    def set_weather_predictions(self, weatherPredictions):
        pass

    def _datapoint_to_dataframe(self, action, t_in):
        """A helper function that converts a datapoint to a pd.df used for predictions.
        Assumes that we have self.zone"""
        X = {"a1": int(0 < action <= 1), "a2": int(1 < action <= 2),
             "t_in": t_in}

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
        pass


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
        X = self._datapoint_to_dataframe(action, t_in)  # TODO which t_in are we really assuming?
        return super(AverageMPCThermalModel, self).predict(X)

    # def save_to_config(self):
    #     """saves the whole model to a yaml file.
    #     RECOMMENDED: PYAML should be installed for prettier config file."""
    #     config_dict = {}
    #
    #     # store zone temperatures
    #     config_dict["Zone Temperatures"] = self.zoneTemperatures
    #
    #     # store coefficients
    #     coefficients = {parameter_name: param for parameter_name, param in
    #                     zip(super(MPCThermalModel, self)._params_order, super(MPCThermalModel, self)._params)}
    #     config_dict["coefficients"] = coefficients
    #
    #     # store evaluations and RMSE's.
    #     config_dict["Evaluations"] = {}
    #     config_dict["Evaluations"]["Baseline"] = super(MPCThermalModel, self).baseline_error
    #     config_dict["Evaluations"]["Model"] = super(MPCThermalModel, self).model_error
    #     config_dict["Evaluations"]["ActionOrder"] = super(MPCThermalModel, self).scoreTypeList
    #     config_dict["Evaluations"]["Better Than Baseline"] = super(MPCThermalModel, self).betterThanBaseline
    #
    #     with open("../ZoneConfigs/thermal_model_" + self.zone, 'wb') as ymlfile:
    #         # TODO Note import pyaml here to get a pretty config file.
    #         try:
    #             import pyaml
    #             pyaml.dump(config_dict[self.zone], ymlfile)
    #         except ImportError:
    #             yaml.dump(config_dict[self.zone], ymlfile)


if __name__ == '__main__':
    import sys

    sys.path.append("..")
    sys.path.append("../MPC")

    from ControllerDataManager import ControllerDataManager
    from xbos import get_client

    # with open("../Buildings/avenal-recreation-center/avenal-recreation-center.yml") as f:
    #     cfg = yaml.load(f)
    #
    # c = get_client()
    #
    # dataManager = ControllerDataManager(cfg, c)
    #
    # all_data = dataManager.thermal_data(days_back=20)
    # with open("Temp_data", "wb") as f:
    #     import pickle
    #     pickle.dump(all_data, f)
    zone = "HVAC_Zone_Large_Room"

    with open("../iPythonNotebooks/Temp_data", "r") as f:
        import pickle
        all_data = pickle.load(f)[zone]



    data = all_data.iloc[-100:]
    thermal_model = ThermalModel()
    thermal_model.fit(all_data, all_data["t_next"])



    def f(row):
        # print(row)
        # res = thermal_model.predict(row.to_frame().T)
        print(row)
        # a = row["t_in"] + 2
        return row




    predicted = data.apply(lambda row: f(row), axis=1)
    print(predicted)
