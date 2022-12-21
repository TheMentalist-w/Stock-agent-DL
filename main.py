from matplotlib import pyplot as plt

import dataPreprocessing as dpp
import dataProcessing as dp
from neuralNetwork import *

if __name__ == '__main__':

    dataframes = dpp.load_data("./data")  # load data from ./data directory
    matrix = dpp.join_dataframes(dataframes)  # join dataframes
    matrix = matrix.drop_duplicates()  # TODO

    # drop NaN values.
    # ,,all" - drop rows that consist of _only_ Nan values
    # ,,any" - drop rows that have _at least_ one Nan value
    matrix = matrix.dropna(how="any")  # TODO

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

    # nerual network part
    # train, test = train_test_split(train, test_size = 0.2, random_state = 42)

    input_dim, input_length = len(train.index), len(train.columns)
    rnn = RNN(input_dim=input_dim, input_length=input_length)

    print(rnn.model.summary())

    rnn.model.fit(train, epochs=1)
    pass
