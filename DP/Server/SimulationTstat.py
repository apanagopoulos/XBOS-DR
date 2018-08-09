import sys

sys.path.append("./MPC/ThermalModels")
sys.path.append("..")

import numpy as np
import utils


# TODO distinguish between actions and add different noise correspondingly.
class SimulationTstat:
    def __init__(self, mpc_thermal_model, curr_temperature):
        self.mpc_thermal_model = mpc_thermal_model
        self.temperature = curr_temperature
        self.time_step = 0
        self.gaussian_distributions = {}

    def set_gaussian_distributions(self, X, y):
        """Needs to be called before calling .temperature. Sets the guassian distribution for the noise
        given by the data X, y.
        :param X: the test_data. pd.df with timeseries data and columns "action", 
        "t_in", "t_out" and "zone_temperatures_*"
        :param y: the expected labels for X. In order of X data.    
        """
        # no action
        rmse, mean_error, std = self.mpc_thermal_model.thermal_model.score(X=X, y=y, scoreType=utils.NO_ACTION)
        self.gaussian_distributions[utils.NO_ACTION] = [mean_error, std]

        # heating
        heating_rmse, heating_mean_error, heating_std = \
            self.mpc_thermal_model.thermal_model.score(X=X, y=y, scoreType=utils.HEATING_ACTION)
        self.gaussian_distributions[utils.HEATING_ACTION] = [heating_mean_error, heating_std]

        # cooling
        cooling_rmse, cooling_mean_error, cooling_std = \
            self.mpc_thermal_model.thermal_model.score(X=X, y=y, scoreType=utils.COOLING_ACTION)
        self.gaussian_distributions[utils.COOLING_ACTION] = [cooling_mean_error, cooling_std]

        # two stage heating
        two_heating_rmse, two_heating_mean_error, two_heating_std = \
            self.mpc_thermal_model.thermal_model.score(X=X, y=y, scoreType=utils.TWO_STAGE_HEATING_ACTION)
        self.gaussian_distributions[utils.TWO_STAGE_HEATING_ACTION] = [two_heating_mean_error, two_heating_std]

        # two stage cooling
        two_cooling_rmse, two_cooling_mean_error, two_cooling_std = \
            self.mpc_thermal_model.thermal_model.score(X=X, y=y, scoreType=utils.TWO_STAGE_COOLING_ACTION)
        self.gaussian_distributions[utils.TWO_STAGE_COOLING_ACTION] = [two_cooling_mean_error, two_cooling_std]



    def next_temperature(self, debug=False):
        """Needs to be called before using the next temperature. 
        Predicts for the given action and adds noise as given by the guassian distribution from the error
        of the thermal model. Also, updates the time_step by one so we know how often we have predicted.
        NOTE: Assumes that mpc_thermal_model has the outside temperatures it used to predict in Advise and the last
        action the Advise predicted as the optimal action.
        :param debug: boolean, whether to return more infromation for debugging. such as returning the noise as well.
        :return int, the current temperature."""
        # inferring the last action from the mpc_thermal_model.
        action = self.mpc_thermal_model.last_action

        # Make sure we trained the gaussian
        try:
            gaussian_mean, gaussian_std = self.gaussian_distributions[action]
        except AttributeError:
            raise RuntimeError("You must train gaussian before predicting data!")

        self.time_step += 1

        # TODO fix the outside temperature. This is not quiet right since the dictinary need not be in order.
        curr_outside_temperature = 70  # self.mpc_thermal_model.outside_temperature.values()[0]
        next_temperature = self.mpc_thermal_model.predict(self.temperature, action,
                                                          outside_temperature=curr_outside_temperature, debug=False)[0]
        noise = np.random.normal(gaussian_mean, gaussian_std)
        self.temperature = next_temperature + noise
        if debug:
            return self.temperature, noise
        else:
            return self.temperature
