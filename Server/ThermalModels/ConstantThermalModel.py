
import numpy as np
import pandas as pd
from ParentThermalModel import ParentThermalModel

import yaml
from scipy.optimize import curve_fit
# daniel imports

import sys
sys.path.append("..")
import utils


# TODO Implement two stage logic better. right now we are just going by whether we heat or not.
class ConstantThermalModel(ParentThermalModel):
    """Thermal model that has constants set for heating, cooling and doing nothing."""

    def __init__(self, thermal_precision=0.05, learning_rate=0.00001):
        '''

        :param thermal_precision: the number of decimal points when predicting.
        '''
        self._params = None
        self._params_order = None
        self._filter_columns = None  # order of columns by which to filter when predicting and fitting data.

        self.thermal_precision = thermal_precision
        self.learning_rate = learning_rate  # TODO evaluate which one is best.

        super(ConstantThermalModel, self).__init__(thermal_precision) # TODO What to do with Learning rate ?


    # ========== average learning ========
    def _features(self, X):
        pass

    # just learning the average of each action and predict with that.
    # thermal model function
    def _func(self, X, *params):
        """The polynomial with which we model the thermal model.
        :param X: np.array with row order (Tin, action)
        :param *coeff: the averages for the thermal model. Should have no_action, heating, cooling as order.
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


        a0_delta, a1_delta, a2_delta = params

        Tin, action = X[0], X[1]

        t_next = Tin + a0_delta * (action == utils.NO_ACTION) + \
                 a1_delta * utils.is_heating(action) + \
                 a2_delta * utils.is_cooling(action)

        return np.array(t_next)

    # Fit without zone interlocking.
    def fit(self, X, y, params=None):
        # TODO how should it update parameters when given more new data?
        """Needs to be called to initally fit the model. Will set self._params to coefficients.
        Will refit the model if called with new data.
        :param X: pd.df with columns ('t_in', 'action')"
        :param y: the labels corresponding to the data. As a pd.dataframe
        :param params: float array [no_action, all heating, all cooling]
        :return self
        """
        filter_columns = ['t_in', 'action']
        # give mapping from params to coefficients and to store the order in which we get the columns.
        self._filter_columns = filter_columns

        assert params is not None
        no_action, heating, cooling = params
        assert cooling < no_action < heating
        self._params = params
        return self


    def update_fit(self, X, y):
        """Adaptive Learning for given datapoints. The data given will all be given the same weight when learning.
        :param X: (pd.df) with columns ('t_in', 'action', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name
        :param y: (float)"""
        pass




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
    # zone = "HVAC_Zone_Large_Room"
    #
    # with open("../iPythonNotebooks/Temp_data", "r") as f:
    #     import pickle
    #     all_data = pickle.load(f)[zone]
    #
    #
    #
    # data = all_data.iloc[-100:]
    # thermal_model = AverageThermalModel()
    # thermal_model.fit(all_data, all_data["t_next"])
    #
    #
    #
    # def f(row):
    #     # print(row)
    #     # res = thermal_model.predict(row.to_frame().T)
    #     print(row)
    #     # a = row["t_in"] + 2
    #     return row
    #
    #
    #
    #
    # predicted = data.apply(lambda row: f(row), axis=1)
    # print(predicted)
    print("AVERAGE THERMAL MODEL")
    import pickle
    building = "avenal-movie-theatre"
    thermal_data = utils.get_data(building=building, days_back=100, evaluate_preprocess=True, force_reload=False)

    model = ConstantThermalModel()
    zone, zone_data = thermal_data.items()[0]
    print(zone)
    model.fit(zone_data, zone_data["t_next"])

    print(model.score(zone_data, zone_data["t_next"]))
    print(model._params)

