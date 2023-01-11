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
    event_container.fill(matrix, corr_thresh=0.98)
    train = event_container.get_train_matrix(event_count_percentage=0.2)  # only events that occured in at least 10% days will be considered

    print("Training dataset:")
    print(train)


    # Just mock data for testing NN, will be deleted later
    """
    import pandas as pd

    xd = []
    for i in range(0, 1000, 1):  # TODO delete it later
        if (i%10) ==1: xd.append([1,1,1,0])
        elif (i%10) ==2: xd.append([1,0,0,0])
        elif (i%10) ==3:xd.append([0,1,1,0])
        else: xd.append([0,0,0,1])

    train = pd.DataFrame(xd)
    print("NEU")
    print(train)
    """

    # splitting train dataset
    # each row in X contains n=steps_back observations from the past,
    # each row in Y contains one observation following these n=steps_back observations from X
    steps_back = 64  # determine how many timestamps you'd like to pass to LSTM model at once
    x, y = dpp.create_x_y_datasets(train, steps_back=steps_back)

    # Build and train nerual network
    # train, test = train_test_split(train, test_size = 0.2, random_state = 42)
    rnn = RNN(n_samples=len(train.index), n_timestamps=steps_back, n_features=len(train.columns))
    print(rnn.model.summary())

    rnn.model.fit(x, y, epochs=10)

    # Get exemplary prediction (list of probabilities of each event)
    corr_probabilities = rnn.predict([x[0:20]])
    corr_names_and_probabs = event_container.probabilities_to_ids_list(corr_probabilities[0], return_top=10)

    print("Predicted correlations in next timestep:")
    for name, p in corr_names_and_probabs:
        print("\t" + "(*) " + name + " with p=" + str(round(100 * p, 2)) + "%")
