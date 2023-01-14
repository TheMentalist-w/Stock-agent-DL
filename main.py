import dataPreprocessing as dpp
import dataProcessing as dp
from neuralNetwork import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from helpers import *
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

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
    bool_matrix = event_container.get_train_matrix(event_count_percentage=0.2)  # only events that occured in at least 10% days will be considered

    print("Binary rolling correlation matrix:")
    print(bool_matrix)

    # create dataframe for every material (bool_matrix + gain(1)/loss(-1))
    trees_data = dict.fromkeys(dataframes.keys(), None)
    for material in trees_data.keys():
        trees_data[material] = dp.binary_gain_loss(bool_matrix, matrix[material])

    # create and train decision trees for every material
    trees = dict.fromkeys(dataframes.keys(), None)
    parameters = {'max_depth': range(5, 15)}
    for tree in trees.keys():
        # X_train, X_test, y_train, y_test = train_test_split(bool_matrix, trees_data[tree], test_size=0.2, shuffle=True,
        #                                          stratify=trees_data[tree])
        print(tree)
        clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
        clf.fit(X=bool_matrix, y=trees_data[tree])
        # tree_model = clf.best_estimator_
        trees[tree] = clf.best_estimator_
        # trees[tree] = DecisionTreeClassifier()
        # trees[tree].fit(X_train, y_train)


    """
    # Just mock data for testing NN, will be deleted later
    import pandas as pd

    xd = []
    for i in range(0, 1000, 1):  # TODO delete it later
        if (i % 10) == 1:
            xd.append([1, 1, 1, 0])
        elif (i % 10) == 2:
            xd.append([1, 0, 0, 0])
        elif (i % 10) == 3:
            xd.append([0, 1, 1, 0])
        else:
            xd.append([0, 0, 0, 1])

    bool_matrix = pd.DataFrame(xd)
    print("Mock matrix:")
    print(bool_matrix)
    """

    # split dataset into train and test parts
    train_matrix, test_matrix = train_test_split(bool_matrix, test_size=0.2, shuffle=False)

    # each row in X contains n=steps_back observations from the past,
    # each row in Y contains one observation following these n=steps_back observations from X
    steps_back = 64  # determine how many timestamps you'd like to pass to LSTM model at once
    X_train, y_train = dpp.create_x_y_datasets(train_matrix, steps_back=steps_back)
    X_test, y_test = dpp.create_x_y_datasets(test_matrix, steps_back=steps_back)

    # Build and train nerual network
    rnn = RNN(n_samples=X_train.shape[0], n_timestamps=X_train.shape[1], n_features=X_train.shape[2])
    print(rnn.model.summary())

    early_stopping = EarlyStopping()  # can be customized, see https://keras.io/api/callbacks/early_stopping/
    history = rnn.model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test)) # TODO callbacks=[early_stopping]
    plot_history(history)

    # Get exemplary prediction (list of probabilities of each event)
    corr_probabilities = rnn.model.predict([X_test[0:20]])
    corr_names_and_probabs = event_container.probabilities_to_ids_list(corr_probabilities[0], return_top=10)

    print("Predicted correlations in next timestep:")
    for name, p in corr_names_and_probabs:
        print("\t" + "(*) " + name + " with p=" + str(round(100 * p, 2)) + "%")

    # threshold for predicted correlation
    thresholded_corr = [1 if p > 0.5 else 0 for name, p in corr_names_and_probabs]
    input = pd.DataFrame([thresholded_corr], columns=bool_matrix.columns)
    print(input)
    gain_los = dict.fromkeys(trees.keys(), None)
    for tree in trees.keys():
        gain_los[tree] = trees[tree].predict(input)
    print(gain_los)
