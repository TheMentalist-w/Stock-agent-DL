import dataPreprocessing as dpp
from matplotlib import pyplot as plt


if __name__ == '__main__':

    dataframes = dpp.load_data("./data")    # load data from ./data directory
    matrix = dpp.join_dataframes(dataframes)  # join dataframes

    # drop NaN values.
    # ,,all" - drop rows that consist of _only_ Nan values
    # ,,any" - drop rows that have _at least_ one Nan value
    matrix = matrix.dropna(how="any")
    matrix = matrix.drop_duplicates() #some days were saved multiple times
    matrix.to_csv('matrix.csv') #saves the matrix for further processing of data
    # print & plot some data!
    print(matrix)

    plt.figure(figsize=(15, 10), dpi=80)

    for col in matrix.columns:
        plt.semilogy(matrix.index, matrix[col], label=col)

    plt.legend()
    plt.savefig("example.png")
