import math
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.spatial import distance


def mins_in_day(timestamp):
    """
    Helper function to calculate how many minutes a certain day has
    """
    return timestamp.hour * 60 + timestamp.minute


def hamming_distance(a, b):
    """
    Calculate the hamming distance
    Parameters
    ----------
    a : ndarray
    b : ndarray
    Returns
    -------
    ndarray
    """
    return distance.hamming(a, b)


def find_similar_days(training_data, now, observation_length, k, method=hamming_distance):
    """
    Calculate and return the k most similar dates with today

    Parameters
    ----------
    training_data : Occupancy Data
    observation_length : Minutes added from previous date, mostly needed for early morning
    k : # of most similar days
    method : similarity method for sorting

    Returns
    -------
    Indexes for the most similar days
    """

    min_time = training_data.index[0] + timedelta(minutes=observation_length)

    # Find moments in our dataset that have the same hour/minute

    selector = ((training_data.index.minute == now.minute) &
                (training_data.index.hour == now.hour) &
                (training_data.index > min_time))

    similar_moments = training_data[selector][:-1]
    obs_td = timedelta(minutes=observation_length)

    # calculate the similarity of each day in the dataset with today
    similar_moments['Similarity'] = [
        method(
            training_data[(training_data.index >= now - obs_td) &
                          (training_data.index <= now)].get_values(),
            training_data[(training_data.index >= i - obs_td) &
                          (training_data.index <= i)].get_values()
        ) for i in similar_moments.index
    ]

    indexes = (similar_moments.sort_values('Similarity', ascending=True).head(k).index)
    return indexes


def predict(data, now, similar_moments, prediction_time, resample_time):
    """
    Using the find_similar_days method, calculate the probability of occupancy
    """
    # initialize the prediction list
    prediction = np.zeros((int(math.ceil(prediction_time / resample_time)) + 1, len(data.columns)))
    # calculate the probability of occupancy
    for i in similar_moments:
        prediction += float(1.0 / float(len(similar_moments))) * data[
            (data.index >= i) & (data.index <= i + timedelta(minutes=prediction_time))]
    # add the known occupancy for the current time to the start of the list
    prediction[0] = data[data.index == now]

    time_index = pd.date_range(now, now + timedelta(minutes=prediction_time), freq=str(resample_time) + 'T')
    return pd.DataFrame(data=prediction, index=time_index)


# TODO find the right number of similar dates and days of data (days of data are in DataManager)
class Occupancy:
    def __init__(self, df, interval, hours, occ_obs_len_addition, sensors):
        self.sensors = sensors
        if sensors:
            observation_length_addition = occ_obs_len_addition * 60  # minutes added from prev date
            k = 5  # number of similar days, not in config - needs validation
            prediction_time = hours * 60  # # of hours ahead for prediction
            resample_time = interval  # interval length
            now = df.index[-1]

            observation_length = mins_in_day(now) + observation_length_addition
            similar_moments = find_similar_days(df, now, observation_length, k)
            self.predictions = predict(df, now, similar_moments, prediction_time, resample_time)
        else:
            self.predictions = df

    def occ(self, now_time):
        """
        Occupancy getter
        """
        if self.sensors:
            return self.predictions[0][now_time]
        else:
            return self.predictions[now_time]


if __name__ == '__main__':
    import yaml
    import sys

    sys.path.insert(0, '..')
    from DataManager import DataManager
    from xbos import get_client

    with open("../config_file.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    with open("../Buildings/ciee/ZoneConfigs/HVAC_Zone_Eastzone.yml", 'r') as ymlfile:
        advise_cfg = yaml.load(ymlfile)

    if cfg["Server"]:
        c = get_client(agent=cfg["Agent_IP"], entity=cfg["Entity_File"])
    else:
        c = get_client()

    dm = DataManager(cfg, advise_cfg, c, "HVAC_Zone_Eastzone")

    occ = Occupancy(dm.preprocess_occ(), 15, 4, 4, advise_cfg["Advise"]["Occupancy_Sensors"])
    for i in range(10):
        print "Intervals ahead: " + str(i)
        print occ.occ(i)
