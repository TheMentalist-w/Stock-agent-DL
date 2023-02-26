from matplotlib import pyplot as plt

import dataPreprocessing as dpp
import dataProcessing as dp
import dataTransformer as dt
import time2Vector as tv
import attention as att
from neuralNetwork import *

import tensorflow as tf
from tensorflow.keras.models import *

from tensorflow.keras.layers import *

batch_size = 32
seq_len = 128
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
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
    dt.Transformer(matrix)
    # preparing data for training
    #event_container = dp.EventContainer()

    df_train,df_val,df_test = dt.Transformer(matrix)
    #event_container.fill(matrix, corr_thresh=0.98)
    #train = event_container.get_train_matrix(event_count_percentage=0.2)  # only events that occured in at least 10% days will be considered
    steps_back = 128 #TODO-find a possibly better num for transformer
    X_train, y_train = dpp.create_x_y_datasets(df_train, steps_back=steps_back)
    X_val, y_val = dpp.create_x_y_datasets(df_val, steps_back=steps_back)
    X_test, y_test = dpp.create_x_y_datasets(df_test, steps_back=steps_back)
    print("Training dataset:")
    print(X_train.shape, y_train.shape)
    #print(train)

    def create_model():
        '''Initialize time and transformer layers'''
        time_embedding = tv.Time2Vector(seq_len)
        attn_layer1 = att.TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer2 = att.TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer3 = att.TransformerEncoder(d_k, d_v, n_heads, ff_dim)

        '''Construct model'''
        in_seq = Input(shape=(seq_len, 8))
        x = time_embedding(in_seq)
        x = Concatenate(axis=-1)([in_seq, x])
        x = attn_layer1((x, x, x))
        x = attn_layer2((x, x, x))
        x = attn_layer3((x, x, x))
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        out = Dense(1, activation='linear')(x)

        model = Model(inputs=in_seq, outputs=out)
        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
        return model


    model = create_model()
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint('time2Vector.py',
                                                  monitor='val_loss',
                                                  save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=1,
                        callbacks=[callback],
                        validation_data=(X_val, y_val))

    model = tf.keras.models.load_model('time2Vector.py',
                                       custom_objects={'Time2Vector': Time2Vector,
                                                       'SingleAttention': SingleAttention,
                                                       'MultiAttention': MultiAttention,
                                                       'TransformerEncoder': TransformerEncoder})

    ###############################################################################
    '''Calculate predictions and metrics'''

    # Calculate predication for training, validation and test data
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Print evaluation metrics for all datasets
    train_eval = model.evaluate(X_train, y_train, verbose=0)
    val_eval = model.evaluate(X_val, y_val, verbose=0)
    test_eval = model.evaluate(X_test, y_test, verbose=0)
    print(' ')
    print('Evaluation metrics')
    print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
    print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
    print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

    ###############################################################################
    '''Display results'''

    fig = plt.figure(figsize=(15, 20))
    st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)
    st.set_y(0.92)

    # Plot training data results
    ax11 = fig.add_subplot(311)
    ax11.plot(train_data[:, 3], label='IBM Closing Returns')
    ax11.plot(np.arange(seq_len, train_pred.shape[0] + seq_len), train_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax11.set_title("Training Data", fontsize=18)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('IBM Closing Returns')
    ax11.legend(loc="best", fontsize=12)

    # Plot validation data results
    ax21 = fig.add_subplot(312)
    ax21.plot(val_data[:, 3], label='IBM Closing Returns')
    ax21.plot(np.arange(seq_len, val_pred.shape[0] + seq_len), val_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax21.set_title("Validation Data", fontsize=18)
    ax21.set_xlabel('Date')
    ax21.set_ylabel('IBM Closing Returns')
    ax21.legend(loc="best", fontsize=12)

    # Plot test data results
    ax31 = fig.add_subplot(313)
    ax31.plot(test_data[:, 3], label='IBM Closing Returns')
    ax31.plot(np.arange(seq_len, test_pred.shape[0] + seq_len), test_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax31.set_title("Test Data", fontsize=18)
    ax31.set_xlabel('Date')
    ax31.set_ylabel('IBM Closing Returns')
    ax31.legend(loc="best", fontsize=12)
    # Just mock data for testing NN, will be deleted later
    """
    import pandas as pd

    xd = []
    for i in range(0, 1000, 1):  # TODO delete it later
        if (i%10) ==1: xd.append([1,0,0,0])
        elif (i%10) ==2: xd.append([0,1,0,0])
        elif (i%10) ==3:xd.append([0,0,1,0])
        else: xd.append([0,0,0,1])

    train = pd.DataFrame(xd)
    print("NEU")
    print(train)
    """
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
    corr_names_and_probabs = event_container.probabilities_to_ids_list(corr_probabilities[0], return_top=3)

    print("Predicted correlations in next timestep:")
    for name, p in corr_names_and_probabs:
        print("\t" + "(*) " + name + " with p=" + str(round(100 * p, 2)) + "%")
"""