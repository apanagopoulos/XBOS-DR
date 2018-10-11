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

        self.past_coeff = []

    # ========== following are function methods for predicting.

    #  $$T_{in} + dt * (T_{in}*a_1*c1 + T_{in}*a_2*c2 + (T_{out} - T_{in})*c_3 + c_4 + \sum_{i = 0}^N (T_{zone_i} - T_{in}) * c_{5+i}$$
    def _features(self, X):
        """Returns the features we are using as a matrix.
        :param X: A matrix with row order (Tin, a1, a2, Tout, dt, rest of zone temperatures)
        :return np.matrix. each row corresponding to a featurized sample in the order of self._param_order"""
        # TODO the order is hardcoded right now.
        # TODO X.shape[1] relies on it being a 2d array.
        # TODO expects numpy array.
        Tin, a1, a2, Tout, dt, zone_temperatures = X[0], X[1], X[2], X[3], X[4], X[5:]
        features = [a1 * Tin, a2 * Tin, Tout - Tin, np.ones(X.shape[1])]  # last entry is for bias
        for zone_temp in zone_temperatures:
            features.append(zone_temp - Tin)
        return dt.T

    # thermal model function
    def _func(self, X, *coeff):
        """The polynomial with which we model the thermal model.
        :param X: np.array with row order (Tin, a1, a2, Tout, dt, rest of zone temperatures)
        :param *coeff: the coefficients for the thermal model. Should be in order: a1, a2, (Tout - Tin),
         bias, zones coeffs (as given by self._params_order)
        """
        # features = self._features(X)
        Tin = X[0]
        dt = X[4]

        self.past_coeff.append(coeff)

        return Tin + coeff # * dt


    # Fit without zone interlocking.
    # TODO train by cutting out small dt's.
    def fit(self, X, y):
        # TODO how should it update parameters when given more new data?
        """Needs to be called to initally fit the model. Will set self._params to coefficients.
        Will refit the model if called with new data.
        :param X: pd.df with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have
        to begin with "zone_temperature_" + "zone name"
        :param y: the labels corresponding to the data. As a pd.dataframe
        :return self
        """
        zone_col = X.columns[["zone_temperature_" in col for col in X.columns]]
        filter_columns = ['t_in', 'a1', 'a2', 't_out', 'dt'] + list(zone_col)

        # give mapping from params to coefficients and to store the order in which we get the columns.
        self._filter_columns = filter_columns
        self._params_order = ["a0"]

        # fit the data. we start our guess with all ones for coefficients.
        # Need to do so to be able to generalize to variable number of zones.
        popt, pcov = curve_fit(self._func, X[filter_columns].T.values, y.values,
                               p0=np.array([6.702505248045286e-05]))

        self._params = np.array(popt)
        return self

    # ===== Function fit methods end. ====

    def updateFit(self, X, y):
        """Adaptive Learning for one datapoint. The data given will all be given the same weight when learning.
        :param X: (pd.df) with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name
        :param y: (float)"""
        # NOTE: Using gradient decent $$self.params = self.param - self.learning_rate * 2 * (self._func(X, *params) - y) * features(X)$$
        loss = self._func(X[self._filter_columns].T.as_matrix(), *self._params)[0] - y
        adjust = self.learning_rate * loss * self._features(X[self._filter_columns].T.as_matrix())
        self._params = self._params - adjust.reshape(
            (adjust.shape[0]))  # to make it the same dimensions as self._params

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
