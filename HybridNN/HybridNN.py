import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from HybridNN_Backend import nnet_initialization

TYPE_CATEGORICAL = 0
TYPE_CONTINUOUS = 1

class ProtectWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_weights=[]):
        self.initial_weights = initial_weights
        return super().__init__()


    def __protect_weights(self):
        weights = self.initial_weights
        for i in range(0, len(weights)):
            new_weights = []
            actual_weights = self.model.layers[i].get_weights()[0]
            for j in range(0, len(actual_weights)):
                new_weights.append([])
                for k in range(0, len(actual_weights[j])):
                    if weights[i][j][k] == 0.0:
                        new_weights[j].append(0.0)
                    else:
                        new_weights[j].append(actual_weights[j][k])
            nnet_layer_weights = [np.array(new_weights), self.model.layers[i].get_weights()[1]]
            self.model.layers[i].set_weights(nnet_layer_weights) 


    def on_epoch_end(self, epoch, logs=None):
        self.__protect_weights()


class HybridNN:
    def __init__(self, 
                 loss='categorical_crossentropy', 
                 optimizer=tf.keras.optimizers.Adam(), 
                 metrics=['accuracy'], 
                 hidden_activation='relu', 
                 output_activation='softmax'):
        self.model = Sequential()
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.nnet_structure = None
        self.df = None


    def init_model(self, df, target=None):
        df, types = self.__prepare_dataset(df, target)
        self.df = df
        arr = df.to_numpy()
        self.nnet_structure = nnet_initialization(arr, types)
        self.__init_structures()


    def __prepare_dataset(self, df, target=None):
        types = []
   
        for i, col in zip(df.dtypes, df):
            if i == 'object' or i == 'bool':
                df[col] = df[col].astype('category').cat.codes
                types.append(TYPE_CATEGORICAL)
            else:
                types.append(TYPE_CONTINUOUS)
        return df, np.array(types)


    def reset_model(self):
        self.model = Sequential()
        self.__init_structures()


    def __init_weights(self):
        weights = self.nnet_structure[0]
        for i in range(0, len(weights)):
           nnet_layer_weights = [np.array(weights[i]), self.model.layers[i].get_weights()[1]]
           self.model.layers[i].set_weights(nnet_layer_weights)


    def __init_structures(self):
        weights = self.nnet_structure[0]
        shape = self.nnet_structure[1]
        self.model.add(Dense(shape[1], input_dim=shape[0], activation=self.hidden_activation))
        for i in range(2, len(weights)):
            self.model.add(Dense(shape[i], activation=self.hidden_activation))
        self.model.add(Dense(shape[len(shape) - 1], activation=self.output_activation))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.__init_weights()


    def __print_weights(self):
        print(self.model.weights)


    def fit(self, *args, **kwargs):
        if kwargs.get('callbacks'):
            if kwargs.get('zero_weight_update') == False:
                kwargs['callbacks'].append(ProtectWeightCallback(self.nnet_structure[0]))
        else:
            if kwargs.get('zero_weight_update') == False:
                kwargs['callbacks'] = [ProtectWeightCallback(self.nnet_structure[0])]
        
        if kwargs.get('zero_weight_update') != None:
            kwargs.pop('zero_weight_update')
        return self.model.fit(*args, **kwargs)


    def get_model(self):
        return self.model
