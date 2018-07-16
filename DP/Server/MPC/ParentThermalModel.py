import sys
sys.path.append("../")

import utils
import numpy as np

from abc import ABCMeta, abstractmethod


class ParentThermalModel:
    """A Parent to the thermal model class which implements all the logic to score and predict."""

    __metaclass__ = ABCMeta

    def __init__(self, thermal_precision):
        self.thermal_precision = thermal_precision

    @abstractmethod
    def _features(self, X):
        """Returns the features we are using as a matrix.
        :param X: A matrix with column order (Tin, a1, a2, Tout, dt, rest of zone temperatures)
        :return np.matrix. each column corresponding to the features in the order of self._param_order"""
        pass

    @abstractmethod
    def _func(self, X, *params):
        """The method which computes predictions given data.
        :param X: pd.df with columns (Tin, action, Tout, dt, rest of zone temperatures)
        :param *params: coefficients in the order of childs self._param_order field. If no coeff given, will use
                 child self._params.
        :return predictions as np.array        
        """
        pass

    @abstractmethod
    def fit(self, X, y, params=None):
        """Needs to be called to initally fit the model. Will set self._params to coefficients.
        Will refit the model if called with new data.
        :param X: pd.df with columns ('t_in', 'action', 't_out', 'dt') and all zone temperature where all have
        to begin with "zone_temperature_" + "zone name"
        :param y: the labels corresponding to the data. As a pd.dataframe
        :param params: Provide it with the parameters to use/guess. e.g. in constantThermalModel this will be use
        to fit the model 
        :return self
        """
        pass

    @abstractmethod
    def update_fit(self, X, y):
        """Adaptive Learning for given datapoints. The data given will all be given the same weight when learning.
        :param X: (pd.df) with columns ('t_in', 'action', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name
        :param y: (float)
        """
        pass

        # TODO. Give this class a field which tells it what the standard interval prediction is. And given dt to predict
        # TODO continiously predict to nearest multiple.
    def predict(self, X, should_round=True):
        """Predicts the temperatures for each row in X.
        :param X: pd.df/pd.Series with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperatures where all 
        have to begin with "zone_temperature_" + "zone name"
        :param should_round: bool. Wether to round the prediction according to self.thermal_precision.
        :return (np.array) entry corresponding to prediction of row in X.
        """
        # only predicts next temperatures

        predictions = self._func(X)


        if should_round:
            # source for rounding: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
            predictions = utils.round_increment(predictions, self.thermal_precision)
        else:
            predictions = predictions

        # consistancy check. Hard coded.
        actions = X["action"]
        inside_temperature = X["t_in"]
        for i in range(len(predictions)):
            pred = predictions[i]
            action = actions[i]
            tin = inside_temperature[i]
            if action == utils.HEATING_ACTION or action == utils.TWO_STAGE_HEATING_ACTION:
                if pred <= tin:
                    pred = tin + self.thermal_precision
            elif action == utils.COOLING_ACTION or action == utils.TWO_STAGE_COOLING_ACTION:
                if pred >= tin:
                    pred = tin - self.thermal_precision

            predictions[i] = pred

        return predictions

    def _RMSE_STD(self, prediction, y):
        '''Computes the RMSE and mean and std of Error.
        NOTE: Use method if you already have predictions.'''
        diff = prediction - y

        mean_error = np.mean(diff)
        rmse = np.sqrt(np.mean(np.square(diff)))
        # standard deviation of the error
        diff_std = np.sqrt(np.mean(np.square(diff - mean_error)))
        return rmse, mean_error, diff_std

    def score(self, X, y, scoreType=-1):
        """Scores the model on the dataset given by X and y.
        :param X: the test_data. pd.df with timeseries data and columns "action", 
        "t_in", "t_out" and "zone_temperatures_*"
        :param y: the expected labels for X. In order of X data.
        :param scoreType: All score on the subset of actions that equal scoreType.
        :returns (floats) rmse, mean, std
        """
        # TODO add capabilities to evaluate all heating and all cooling.
        # Filters data to score only on subset of actions.
        assert scoreType in list(range(-1, 6))
        # filter by the action we want to score by
        if scoreType != -1:
            filter_arr = (X['action'] == scoreType)
        elif scoreType == -1:
            filter_arr = np.ones(X['action'].shape) == 1
        X = X[filter_arr]
        y = y[filter_arr]

        # Predict on filtered data
        prediction = self.predict(X)  # only need to predict for relevant actions

        # Get model error
        rmse, mean_error, std = self._RMSE_STD(prediction, y)

        return rmse, mean_error, std