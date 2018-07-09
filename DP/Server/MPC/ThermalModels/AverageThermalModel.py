
import numpy as np
import pandas as pd
from MPC.ParentThermalModel import ParentThermalModel

import yaml
from scipy.optimize import curve_fit
# daniel imports

import sys
sys.path.append("..")
import utils


# used to find best thermal model.
class AverageThermalModel(ParentThermalModel):

    def __init__(self, thermal_precision=0.05, learning_rate=0.00001):
        '''

        :param thermal_precision: the number of decimal points when predicting.
        '''

        self._params = None
        self._params_order = None
        self._filter_columns = None  # order of columns by which to filter when predicting and fitting data.

        self.thermal_precision = thermal_precision
        self.learning_rate = learning_rate  # TODO evaluate which one is best.

        super(AverageThermalModel, self).__init__(thermal_precision) # TODO What to do with Learning rate ?


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
    def fit(self, X, y):
        # TODO how should it update parameters when given more new data?
        """Needs to be called to initally fit the model. Will set self._params to coefficients.
        Will refit the model if called with new data.
        :param X: pd.df with columns ('t_in', 'action')"
        :param y: the labels corresponding to the data. As a pd.dataframe
        :return self
        """

        filter_columns = ['t_in', 'action']
        # give mapping from params to coefficients and to store the order in which we get the columns.
        self._filter_columns = filter_columns

        def get_delta_mean(action_data):
            # get the mean change of temperature from now to next. Normalized
            # TODO assuming action_data has "t_next"
            # TODO assuming that mean will be a float
            return np.mean(y - action_data["t_in"])

        cooling_data = X[utils.is_cooling(X["action"])]
        heating_data = X[utils.is_heating(X["action"])]
        no_action_data = X[X["action"] == utils.NO_ACTION]

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


    def update_fit(self, X, y):
        """Adaptive Learning for given datapoints. The data given will all be given the same weight when learning.
        :param X: (pd.df) with columns ('t_in', 'a1', 'a2', 't_out', 'dt') and all zone temperature where all have 
        to begin with "zone_temperature_" + "zone name
        :param y: (float)"""
        pass

    # ==== average model ends. ======


class AverageMPCThermalModel(AverageThermalModel):
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

    model = AverageThermalModel()
    zone, zone_data = thermal_data.items()[0]
    print(zone)
    model.fit(zone_data, zone_data["t_next"])

    print(model.score(zone_data, zone_data["t_next"]))
    print(model._params)

