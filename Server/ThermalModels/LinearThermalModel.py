import sys

import numpy as np
import pandas as pd

sys.path.append("..")
sys.path.append("../..")
import utils
from ParentThermalModel import ParentThermalModel

from scipy.optimize import curve_fit


# following model also works as a sklearn model.
class LinearThermalModel(ParentThermalModel):
    def __init__(self, interval_thermal, thermal_precision=0.05, learning_rate=0.00001):
        '''
        :param interval_thermal: The minutes the thermal model learns to predict for. The user is responsible to ensure
                                that the data the model receives for training is as specified. 
        :param thermal_precision: the closest multiple of which to round to.
        '''

        self._params = None
        self._params_coeff_order = None  # first part of _params is coeff part
        self._params_bias_order = None  # the rest is bias part.
        self._filter_columns = None  # order of columns by which to filter when predicting and fitting data.

        self.thermal_precision = thermal_precision
        self.learning_rate = learning_rate  # TODO evaluate which one is best.

        # Set the parent variables
        super(LinearThermalModel, self).__init__(thermal_precision, interval_thermal)


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
                raise RuntimeError("You must train classifier before predicting data!")
            params = self._params

        coeffs = params[:len(self._params_coeff_order)]
        biases = [0, 0, 0, 0] + list(params[len(self._params_coeff_order):])

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
        features = [np.ones(X.shape[1]),  # action == utils.HEATING_ACTION
                    np.ones(X.shape[1]),  # action == utils.COOLING_ACTION
                    np.ones(X.shape[1]),  # action == utils.TWO_STAGE_HEATING_ACTION
                    np.ones(X.shape[1]),  # action == utils.TWO_STAGE_COOLING_ACTION
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

        self._params_bias_order = ['t_out', 'bias'] + list(zone_col)

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







if __name__ == '__main__':
    print("Thermal Model")
    import sys


    sys.path.append("./ThermalModels")
    from AverageThermalModel import AverageThermalModel
    building = "ciee"
    thermal_data = utils.get_data(building=building, days_back=100, evaluate_preprocess=False, force_reload=False)

    model = LinearThermalModel(interval_thermal=5)
    avg_model = AverageThermalModel(interval_thermal=5)
    zone, zone_data = thermal_data.items()[0]

    zone_data = zone_data[zone_data["dt"] == 5]


    # zone_data = zone_data[zone_data["t_min"] != zone_data["t_max"]]
    #
    # zone_data = zone_data[((zone_data["t_in"] > zone_data["t_next"]) | (zone_data["action"] != utils.COOLING_ACTION))]
    #
    # cooling_data = zone_data[(zone_data["t_in"] > zone_data["t_next"]) & (zone_data["action"] == utils.COOLING_ACTION)]

    # with open("weird", "wb") as f:
    #     pickle.dump(zone_data, f)

    print(zone)
    model.fit(zone_data, zone_data["t_next"])
    avg_model.fit(zone_data, zone_data["t_next"])

    # print(model._params)

    # for i in range(-1, 6):
    #     print("normal", model.score(zone_data, zone_data["t_next"], scoreType=i))
    #     print("avg", avg_model.score(zone_data, zone_data["t_next"], scoreType=i))

    print("coeff", model._params[:len(model._params_coeff_order)])
    print("bias", model._params[len(model._params_coeff_order):])

    print(model._params_coeff_order)
    print(model._params_bias_order)

    print("avg coeff", avg_model._params)

    filter_data = utils.prediction_test(zone_data, thermal_model=model)
    # Get all insensible predictions.
    filter_data = filter_data == 0
    print("Number of data", zone_data.shape[0])
    print("Number of insensible", sum(filter_data))
    print("Inconsistent", zone_data[filter_data])
    # print("Predicted", model.predict(zone_data[filter_data]) - zone_data[filter_data]["t_in"])

    new_data = zone_data[filter_data]

    #heating
    new_data["action"] = np.ones(sum(filter_data))
    heating_predict = model.predict(new_data, should_round=False) - new_data["t_in"]
    # cooling
    new_data["action"] = np.ones(sum(filter_data))*2
    cooling_predict = model.predict(new_data, should_round=False) - new_data["t_in"]
    # no action
    new_data["action"] = np.ones(sum(filter_data))*0
    no_predict = model.predict(new_data, should_round=False) - new_data["t_in"]

    print(pd.DataFrame({"heating":heating_predict, "cooling":cooling_predict, "no action": no_predict}))


    #(Tin, action, Tout, dt, rest of zone temperatures)
    X = np.array([[75, 0, 75, 5, 75, 75, 75]]).T
    print(model.predict(X, should_round=False))
    # print(model._func(X))
    # print(model.predict(X))


    # r = therm_data["HVAC_Zone_Shelter_Corridor"].iloc[-1]
    # print(r)
    # print model.predict(t_in=r["t_in"], zone="HVAC_Zone_Shelter_Corridor", action=r["action"],outside_temperature=r["t_out"], interval=r["dt"])
    # print("hi")
