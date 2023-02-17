import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
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

def Transformer(df):
    for d in df:
        #print(d)
        df[d] = df[d].pct_change()  # Create arithmetic returns column

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
    print("Createing training, validation and test split")

    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]


    # Convert pandas columns into arrays
    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values
    print('Training data shape: {}'.format(train_data.shape))
    print('Validation data shape: {}'.format(val_data.shape))
    print('Test data shape: {}'.format(test_data.shape))

    df_train.head()

