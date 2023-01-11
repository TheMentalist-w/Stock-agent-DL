from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


class RNN:
    def __init__(self, n_samples, n_timestamps, n_features):

        self.model = Sequential()

        # Recurrent layer
        self.model.add(LSTM(units=128,
                            input_shape=(n_timestamps, n_features),
                            return_sequences=False,
                            dropout=0.1, recurrent_dropout=0.1))

        # Fully connected layer
        self.model.add(Dense(256, activation='relu'))

        # Dropout for regularization
        self.model.add(Dropout(0.5))

        # Output layer
        self.model.add(Dense(n_features, activation='sigmoid'))  # self.model.add(Dense(n_features, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        # TODO add earlystopping layer


    def predict(self, sequence):
        return self.model.predict(sequence)
