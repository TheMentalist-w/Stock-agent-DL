import itertools
import pandas as pd
import numpy as np
from collections import defaultdict


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

        res_sum=defaultdict(float)
        for event in filtered_events:
            res[event.ID][event.date]=1

        # handle NO_EVENT case
        res["NO_EVENT"] = res.apply(lambda row: 0 if any(row) else 1).astype(bool)
        res["sum"]=0
        res["sum_abs"] = 0.0
        for event in filtered_events:
            res["sum"][event.date]=event.Corr+res["sum"]
            res["sum_abs"][event.date]=abs(event.Corr)+res["sum_abs"]
        #res.append(pd.Series(res_sum))

        # fill one-hot to ids vocabulary
        for i, index in enumerate(res.columns):
            self.one_hot_to_ids[i] = index

        return res.fillna(0)

    def probabilities_to_ids_list(self, input_list, return_top=5):
        ids_and_probs = [(self.one_hot_to_ids[i], value) for i, value in enumerate(input_list)]
        return list(sorted(ids_and_probs, key=lambda x: x[1], reverse=True))[:return_top]


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
