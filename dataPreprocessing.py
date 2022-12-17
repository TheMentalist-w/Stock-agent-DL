import glob
import pandas as pd
import os
import re


def load_data(path="./data"):
    """

    :param path: path to directory with input files
    :return: dictionary containing imported data (dataframes)

    """

    filepaths = glob.glob(f"{path}/*.csv")

    dataframes = {}

    for filepath in filepaths:
        data = pd.read_csv(filepath)

        if {"Data", "Zamkniecie"}.issubset(data.columns):
            data = data[["Data", "Zamkniecie"]]
            data["Data"] = pd.to_datetime(data["Data"])
            data = data.rename(columns={'Data': 'Date', 'Zamkniecie': 'Value'})

        elif {"Data", "Ostatnio"}.issubset(data.columns):
            data = data[["Data", "Ostatnio"]]
            data["Data"] = pd.to_datetime(data["Data"], format="%d.%m.%Y")
            data = data.rename(columns={'Data': 'Date', 'Ostatnio': 'Value'})

        elif {"Date", "Price"}.issubset(data.columns):
            data = data[["Date", "Price"]]
            data["Date"] = pd.to_datetime(data["Date"])
            data = data.rename(columns={'Date': 'Date', 'Price': 'Value'})

        else:
            raise NotImplementedError(f"Unsupported data format in file {filepath}!")

        filename = re.split("[\.#]", os.path.basename(filepath))[0]

        if data["Value"].dtype != float:
            data["Value"] = data["Value"].apply(lambda x: float(x.replace('.', '').replace(',', '.')))

        if filename not in dataframes.keys():
            dataframes[filename] = data
        else:
            dataframes[filename] = pd.concat([dataframes[filename], data], ignore_index=True)

    return dataframes


def join_dataframes(dataframes_dict):
    """

    :param dataframes_dict: dictionary with dataframes
    :return: matrix containing joined dataframes

    """

    all_dates = pd.concat([df["Date"] for df in dataframes_dict.values()])
    min_date, max_date = min(all_dates), max(all_dates)

    res_df = pd.DataFrame({'Date': pd.date_range(min_date, max_date)})

    for name, dataframe in dataframes_dict.items():
        dataframe = dataframe.rename(columns={'Value': name})
        res_df = res_df.merge(dataframe, on="Date", how="left")

    res_df.set_index(["Date"], inplace=True)

    return res_df

