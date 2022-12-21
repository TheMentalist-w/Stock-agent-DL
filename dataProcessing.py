import itertools
import pandas as pd
import numpy as np


class EventContainer:
    def __init__(self):
        self.container = []

    def add_event(self, event):
        self.container.append(event)

    def get_available_values(self, which):
        if which == "event_ids":
            return set([event.ID for event in self.container])
        elif which == "event_dates":
            return sorted(set([event.date for event in self.container]))
        elif which == "events":
            return set([event for event in self.container])
        else:
            raise AssertionError

    def fill(self, matrix, corr_thresh=0.95):

        for var_1, var_2 in set(itertools.product(matrix, matrix)):  # iterate over all combinations of variables

            if var_1 == var_2:
                continue

            # calculate rolling correlation
            rolling_corr = matrix[var_1].rolling('5d', min_periods=1).corr(matrix[var_2])

            # get dates when corr between var_1 and var_2 occured
            # (only values  bigger than corr_threshold are taken into consideration)
            rolling_corr_dates = rolling_corr[abs(rolling_corr) > corr_thresh].index

            for date in rolling_corr_dates:  # create events and add them to container for further processing
                event = Event(var_1, var_2, date)
                self.add_event(event)

    def get_train_matrix(self, event_count_thresh = 10):

        # create training matrix; in columns there are dates, in rows '1' if givren event occurred, '0' otherwise
        res = pd.DataFrame(
            columns=self.get_available_values("event_dates"),
            index=self.get_available_values("event_ids")
        )

        ids = [event.ID for event in self.get_available_values("events")]
        freqs = dict(zip(*np.unique(ids, return_counts=True)))

        # contains only events that occurred at least [event_count_thresh times+1] times in dataset
        filtered_events = filter(lambda x: freqs[x.ID] > event_count_thresh, self.get_available_values("events"))

        for event in filtered_events:
            res[event.date][event.ID] = 1

        return res.fillna(0)


class Event:

    # class for storing ,,events" observed in data, for example "price of gold correlated with price of gas on 12-11-2022"
    def __init__(self, var_1, var_2, date):
        self.var_1, self.var_2 = var_1, var_2
        self.date = date

    @property
    def ID(self):
        return self.var_1 + ";" + self.var_2
