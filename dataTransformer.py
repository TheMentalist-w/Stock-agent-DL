import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
import itertools
from collections import defaultdict
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

batch_size = 32
seq_len = 128
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
class Event:

    # class for storing ,,events" observed in data, for example "price of gold correlated with price of gas on 12-11-2022"
    def __init__(self, var_1, var_2, date,corr=0):
        self.var_1, self.var_2 = var_1, var_2
        self.date = date
        self.corr=corr

    @property
    def ID(self):
        return self.var_1 + ";" + self.var_2

    @property
    def Corr(self):
        return self.corr


class EventContainer:
    def __init__(self):
        self.container = []
        self.one_hot_to_ids = {}

    def add_event(self, event):
        self.container.append(event)

    def get_available_values(self, which):
        if which == "event_ids":
            return set([event.ID for event in self.container])
        elif which == "event_dates":
            return sorted(set([event.date for event in self.container]))
        elif which == "event_corr":
            return sorted(set([event.corr for event in self.container]))
        elif which == "events":

            # return sorted(set([(event.var_1, event.var_2, event.corr, event.date) for event in self.container]))
            return set([event for event in self.container])
        else:
            raise AssertionError
    def fill(self, matrix, corr_thresh=0.98):

        for var_1, var_2 in itertools.combinations(matrix.columns, 2):  # iterate over all combinations of variables

            # calculate rolling correlation
            rolling_corr = matrix[var_1].rolling('5d', min_periods=1).corr(matrix[var_2])

            # get dates when corr between var_1 and var_2 occured
            # (only values  bigger than corr_threshold are taken into consideration)
            rolling_corr_dates = rolling_corr[abs(rolling_corr) > corr_thresh].index
            for date in rolling_corr_dates:  # create events and add them to container for further processing
                event = Event(var_1, var_2, date,rolling_corr[date])
                self.add_event(event)

    def get_train_matrix(self, event_count_percentage=0.3):

            # create training matrix; in columns there are dates, in rows '1' if given event occurred, '0' otherwise

            res = pd.DataFrame(
                index=self.get_available_values("event_dates"),
                columns=self.get_available_values("event_ids")
            )

            ids = [event.ID for event in self.get_available_values("events")]
            freqs = dict(zip(*np.unique(ids, return_counts=True)))

            # contains only events that occurred at least [event_count_thresh times+1] times in dataset
            event_count_thresh = len(self.get_available_values("event_dates")) * event_count_percentage
            filtered_events = filter(lambda x: freqs[x.ID] > event_count_thresh, self.get_available_values("events"))

            res_sum = defaultdict(float)
            for event in filtered_events:
                res[event.ID][event.date] = 1

            # handle NO_EVENT case
            res["NO_EVENT"] = res.apply(lambda row: 0 if any(row) else 1).astype(bool)
            res["sum"] = 0
            res["sum_abs"] = 0.0
            for event in filtered_events:
                res["sum"][event.date] = event.Corr + res["sum"]
                res["sum_abs"][event.date] = abs(event.Corr) + res["sum_abs"]
            # res.append(pd.Series(res_sum))

            # fill one-hot to ids vocabulary
            for i, index in enumerate(res.columns):
                self.one_hot_to_ids[i] = index

            return res.fillna(0)

    def probabilities_to_ids_list(self, input_list, return_top=5):
            ids_and_probs = [(self.one_hot_to_ids[i], value) for i, value in enumerate(input_list)]
            return list(sorted(ids_and_probs, key=lambda x: x[1], reverse=True))[:return_top]

def Transformer(df):
#for d in df:
        #print(d) df[d] = df[d].pct_change()  # Create arithmetic returns column
    event_container=EventContainer()
    for var_1, var_2 in itertools.combinations(df.columns, 2):  # iterate over all combinations of variables
      corr_thresh=0.98
    # calculate rolling correlation
      rolling_corr = df[var_1].rolling('5d', min_periods=1).corr(df[var_2])

      # get dates when corr between var_1 and var_2 occured
      # (only values  bigger than corr_threshold are taken into consideration)
      rolling_corr_dates = rolling_corr[abs(rolling_corr) > corr_thresh].index
      for date in rolling_corr_dates:  # create events and add them to container for further processing
          event = Event(var_1, var_2, date,rolling_corr[date])
          event_container.add_event(event)
    df.dropna(how='any', axis=0, inplace=True)  # Drop all rows with NaN values

    ###############################################################################
    print("Createing indexes to split dataset")

    times = len(df)
    last_10pct = sorted(df.index.values)[-int(0.1 * times)]  # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2 * times)]  # Last 20% of series

    ###############################################################################
    print("Normalizing price columns")
    #
    min_return = min(df[(df.index < last_20pct)].min(axis=0))
    max_return = max(df[(df.index < last_20pct)].max(axis=0))

    # Min-max normalize price columns (0-1 range)

    for d in df:
        df[d] = (df[d] - min_return) / (max_return - min_return)

    ###############################################################################
    print("Split the data into training, validation and test data.")

    df_train = df[(df.index < last_20pct)]  # Training data are eighty percent of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)] #ten percent for validation
    df_test = df[(df.index >= last_10pct)] #ten percent for testing


    # Convert pandas columns into arrays
    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values
    print(f'Shapes of training:{format(train_data.shape)},validation{format(val_data.shape)}, and test {format(test_data.shape)} data')
    df_train.head()
    return df_train, df_val, df_test