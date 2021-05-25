import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from HybridNN_Backend import nnet_initialization
from keras.optimizers import SGD

TYPE_CATEGORICAL = 0
TYPE_CONTINUOUS = 1


class HybridNN:
    def __init__(self, loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'], hidden_activation='relu', output_activation='softmax'):
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

    def __init_weights(self):
        weights = self.nnet_structure[0]
        for i in range(0, len(weights)):
           nnet_layer_weights = [np.array(weights[i]), self.model.layers[i].get_weights()[1]]
           self.model.layers[i].set_weights(nnet_layer_weights)
           #print(self.model.layers[i].get_weights())

    def __init_structures(self):
        weights = self.nnet_structure[0]
        shape = self.nnet_structure[1]
        self.model.add(Dense(shape[1], input_dim=shape[0], activation=self.hidden_activation))
        for i in range(2, len(weights)):
            self.model.add(Dense(shape[i], activation=self.output_activation))
        self.model.add(Dense(shape[len(shape) - 1], activation=self.output_activation))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.__init_weights()

    def __loss(self, x, y, loss_object):
        y_ = self.model(x, training=True)
        return loss_object(y_true=y, y_pred=y_)

    def __grad(self, inputs, targets, loss_object):
        with tf.GradientTape() as tape:
            loss_value = self.__loss(inputs, targets, loss_object)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, epochs=100, train_dataset=None, batch_size=10):
        train_dataset = train_dataset.batch(batch_size)
        loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                loss_value, grads = self.__grad(x_batch_train, y_batch_train, loss_object)

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(y_batch_train, self.model(x_batch_train, training=True))
               
            if epoch % 10 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
    
            #self.__init_weights()


    def get_model(self):
        return self.model


if __name__ == '__main__':
    df = pd.read_csv('train.csv', delimiter=',')
    df.drop('index', inplace=True, axis=1)
    df.drop('id', inplace=True, axis=1)
    print("-----------")
    hybridNN = HybridNN()
    hybridNN.init_model(df, target='satisfaction')



    hybridNN.get_model().fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)