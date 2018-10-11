import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
# daniel imports
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


def cross_rmse_actions(bldg, action=-1):
    """Evaluates the thermal model through cross validation. 
    _params:
        action: (int) The action by which to pre-filter training data and for which to find RMSE. -1 indicates no filter, 0 no action,
                    1 heating, 2 cooling.
    returns:
        (int) RMSE for each fold of cross validation
    """
    print("start cross validation")
    # get data
    import pickle
    z_file = open("zone_thermal_" + bldg)
    data = pickle.load(z_file)
    z_file.close()
    for zone, thermal_data in data.items():
        ThermalFit.trivial_rmse = []
        thermal = ThermalFit(scoreType=-1)  # change score type here.
        if action == 0:
            X = thermal_data[(thermal_data['a1'] == 0) & (thermal_data['a2'] == 0)]
        elif action == 1:
            X = thermal_data[thermal_data['a1'] == 1]
        elif action == 2:
            X = thermal_data[thermal_data['a2'] == 1]
        else:
            X = thermal_data
        print("training data shape: ", X.shape)
        X = shuffle(X)  # to get an even spread of all actions.
        y = X['t_next']
        real_rmse = cross_val_score(thermal, X, y)
        print("real_rsme", real_rmse)
        trivial_rmse = ThermalModel.trivial_rmse
        print("trivial rmse", trivial_rmse)
        diff = np.mean(real_rmse) - np.mean(trivial_rmse)
        print("diff in trival and normal rmse (costum.rmse - trivial)", diff)

        return {"building": bldg, "zone": zone, "real_rmse": real_rmse, "trival_rmse": trivial_rmse,
                "(costum.rmse - trivial)": diff,
                "num_data": X.shape, "num_folds": 3, "action": action, "timeframe": "01/01/18-03/01/18"}


class ThermalModel(BaseEstimator, RegressorMixin):
    # keeping track of all the rmse's computed with this model
    trivial_rmse = []
    rmse = []
    scoreTypeList = []  # to know which action each rmse belongs to.

    def __init__(self, scoreType=-1):
        '''
        _params:
            scoreType: (int) which actions to filter by when scoring. -1 indicates no filter, 0 no action,
                        1 heating, 2 cooling.
        '''
        self.scoreType = scoreType

        self._params = None

    # thermal model function
    def _func(self, X, *args):
        Tin, a1, a2, Tout, dt, zone_temperatures = X[0], X[1], X[2], X[3], X[4], X[5:]

        c1, c2, c3, c4, c_rest = args[0], args[1], args[2], args[3], args[4:]
        return Tin + (c1 * a1 * Tin + c2 * a2 * Tin + c3 * (Tout - Tin) + c4 +
                      sum([c * (zone_temp - Tin) for c, zone_temp in zip(c_rest, zone_temperatures)])) * dt

    def fit(self, X, y=None):
        zone_col = X.columns[["temperature_" in col for col in X.columns]]
        filter_columns = ['t_in', 'a1', 'a2', 't_out', 'dt'] + list(zone_col)

        popt, pcov = curve_fit(self._func, X[filter_columns].T.as_matrix(), y.as_matrix(),
                               p0=np.ones(4 + len(zone_col)))  # fit the data
        self._params = popt
        return self

    def predict(self, X, y=None):
        # only predicts next temperatures
        try:
            getattr(self, "_params")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        zone_col = X.columns[["temperature_" in col for col in X.columns]]
        filter_columns = ['t_in', 'a1', 'a2', 't_out', 'dt'] + list(zone_col)

        res = [self._func(X.loc[date][filter_columns], *self._params)
               for date in X.index]

        return res

    def _normalizedRMSE(self, dt, prediction, y):
        '''Computes the RMSE with scaled differences to normalize to 15 min intervals.'''
        diff = prediction - y
        diff_scaled = diff * 15. / dt  # to offset for actions which were less than 15 min. makes everything a lot worse
        rmse = np.sqrt(np.mean(np.square(diff_scaled)))
        return rmse

    def score(self, X, y, sample_weight=None):
        ThermalFit.scoreTypeList.append(self.scoreType)

        if self.scoreType == 0:
            filter_arr = (X['a1'] == 0) & (X['a2'] == 0)
        elif self.scoreType == 1:
            filter_arr = X['a1'] == 1
        elif self.scoreType == 2:
            filter_arr = X['a2'] == 1
        else:
            filter_arr = np.ones(X['a1'].shape) == 1

        X = X[filter_arr]
        y = y[filter_arr]

        prediction = self.predict(X)  # only need to predict for relevant actions

        rmse = self._normalizedRMSE(X['dt'], prediction, y)

        # add normal RMSE for reference.
        ThermalFit.rmse.append(rmse)

        # add trivial error for reference.
        trivial_rmse = self._normalizedRMSE(X['dt'], X['t_in'], y)
        ThermalFit.trivial_rmse.append(trivial_rmse)

        return rmse


if __name__ == "__main__":

    bldg_rmse_data = []
    bldgs = ["north-berkeley-senior-center", "orinda-community-center", "south-berkeley-senior-center",
             "word-of-faith-cc", "hayward-station-8"]
    for bldg in bldgs:
        for i in range(-1, 3):
            print("Curr building:", bldg)
            print("Curr action:", i)
            try:
                data = cross_rmse_actions(bldg, i)
                bldg_rmse_data.append(data)
            except:
                continue
    df = pd.DataFrame(bldg_rmse_data)
# df.to_csv('rmse_buildings_old_model.csv')
