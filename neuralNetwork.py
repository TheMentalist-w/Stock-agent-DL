# ORIGINAL SOURCE: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
# just a basic template - to be modified

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding


class RNN:
    def __init__(self, input_dim, input_length):

        self.model = Sequential()

        # Embedding layer
        self.model.add(
            Embedding(input_dim = input_dim,            # rozmiar słownika eventów
                      input_length = input_length,      # liczba przypadków uczących
                      output_dim=100,                   # wymiarowość embeddingu
                      trainable=False,                  # weights=[embedding_matrix] & mask_zero=True  wyrzuciłem, trainable = False, ponieważ -> https://datascience.stackexchange.com/questions/67801/using-trainable-true-in-keras-embedding-obtained-better-performance
                      ))

        # Masking layer for pre-trained embeddings
        self.model.add(Masking(mask_value=0.0))

        # Recurrent layer
        self.model.add(LSTM(64, return_sequences=False,
                       dropout=0.1, recurrent_dropout=0.1))

        # Fully connected layer
        self.model.add(Dense(64, activation='relu'))

        # Dropout for regularization
        self.model.add(Dropout(0.5))

        # Output layer
        self.model.add(Dense(input_dim, activation='softmax'))

        # Compile the model
        self.model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

