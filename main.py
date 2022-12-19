import dataPreprocessing as dpp
import dateTimeWindows as dtw
from matplotlib import pyplot as plt
global testing

if __name__ == '__main__':
    testing=False
    dataframes = dpp.load_data("./data")    # load data from ./data directory
    matrix = dpp.join_dataframes(dataframes)  # join dataframes

    # drop NaN values.
    # ,,all" - drop rows that consist of _only_ Nan values
    # ,,any" - drop rows that have _at least_ one Nan value
    matrix = matrix.dropna(how="any")
    matrix = matrix.drop_duplicates() #some days were saved multiple times
    matrix.to_csv('matrix.csv') #saves the matrix for further processing of data
    # print & plot some data!
    if testing: print(matrix)

    plt.figure(figsize=(15, 10), dpi=80)

    for col in matrix.columns:
        plt.semilogy(matrix.index, matrix[col], label=col)

    plt.legend()
    plt.savefig("example.png")

    #further processing
    dtw.load_matrix()
