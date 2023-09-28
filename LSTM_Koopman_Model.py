from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np


class encoder(tf.keras.layers.Layer):
    def __init__(self, nencoded, **kwargs):
        super(encoder, self).__init__(**kwargs)
        self.nencoded = nencoded
        self.lstm = layers.LSTM(self.nencoded, return_sequences=False, return_state=False)
        self.dense = layers.Dense(units=nencoded, activation='relu')

    def call(self, input_lstm, input_dense, input_freq):
        lstm_out = self.lstm(input_lstm)
        dense_out = self.dense(input_dense)
        encoded = lstm_out + dense_out
        combined_encoded = tf.keras.layers.Concatenate()([input_freq, encoded])
        return encoded, combined_encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({"nencoded": self.nencoded})
        return config

class Koopman(tf.keras.layers.Layer):
    def __init__(self, nencoded, nfrq, **kwargs):
        super(Koopman, self).__init__(**kwargs)
        self.nencoded = nencoded
        self.nfrq = nfrq
        self.KoopmanOperator = layers.Dense(units=self.nencoded+self.nfrq, activation=None, use_bias=False)
        
    def call(self, combined_encoded):
        advanced_encoded = self.KoopmanOperator(combined_encoded)
        return advanced_encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({"nencoded": self.nencoded, "nfrq": self.nfrq})
        return config

class decoder(tf.keras.layers.Layer):
    def __init__(self, nfreq, **kwargs):
        super(decoder, self).__init__(**kwargs)
        self.nfreq = nfreq
        
    def call(self, advanced_encoded):
        advanced_freq = advanced_encoded[:, :self.nfreq]
        return advanced_freq
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({"nfreq": self.nfreq})
        return config
