import dataPreprocessing as dpp
import dataProcessing as dp
from neuralNetwork import *
from datetime import date

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    dataframes = dpp.load_data("./data")  # load data from ./data directory
    matrix = dpp.join_dataframes(dataframes)  # join dataframes
    matrix = matrix.drop_duplicates()  # TODO just a temporary solution, not sure why join_dataframes contains duplicates

    # drop NaN values.
    # ,,all" - drop rows that consist of _only_ Nan values
    # ,,any" - drop rows that have _at least_ one Nan value
    matrix = matrix.dropna(how="any")

    # TODO what to do with ,,holes" in dates range, i. e. when a few days is missing in the dataset?
    #  Maybe approximate them?

    days_in_range = (max(matrix.index) - min(matrix.index)).days
    print(f"Whole date range should contain {days_in_range - (days_in_range // 7) * 2} days, "
          f"but contains"f"{len(set(matrix.index))}. This may have negative impact on training process :|")

    # print & plot some data!
    print(matrix)

    plt.figure(figsize=(15, 10), dpi=80)

    for col in matrix.columns:
        plt.semilogy(matrix.index, matrix[col], label=col)

    plt.legend()
    plt.savefig("example.png")

    # preparing data for training
    event_container = dp.EventContainer()
    event_container.fill(matrix)
    train = event_container.get_train_matrix()

    print("Training dataset:")
    print(train)

    # creating nerual network
    # train, test = train_test_split(train, test_size = 0.2, random_state = 42)

    input_dim = len(train.index)         # len( event_container.get_available_values("event_ids") )
    input_length = len(train.columns)    # len( event_container.get_available_values("events") )
    print(f"input_dim = {input_dim}, input_length={input_length}")

    rnn = RNN(input_dim = input_dim, input_length=input_length)
    print(rnn.model.summary())

    rnn.model.fit( train, epochs=1 )

    print(rnn.model.summary())
    pass
