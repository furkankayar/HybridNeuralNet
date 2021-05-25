import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from HybridNN import HybridNN
from keras.optimizers import SGD

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='sigmoid'),
    tf.keras.layers.Dense(4, activation='sigmoid'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model


def iris():
    df = pd.read_csv('iris.csv', delimiter=',')
    print("-----------")
    opt = SGD(lr=0.01)
    hybridNN = HybridNN(hidden_activation='sigmoid')
    hybridNN.init_model(df, target='Species')
    
    labels = hybridNN.df['Species']
    features = hybridNN.df.iloc[:, 0:4]

    X = features 

    encoder = LabelEncoder()
    y = np.ravel(labels)
    #encoder.fit(y)
    #y = encoder.transform(y)
    #y = np_utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(len(X_train))
  
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    hybridNN.train(train_dataset = train_dataset, epochs = 1000000)

    """
    overfitCallback = EarlyStopping(monitor='val_loss', patience=100)
    model = hybridNN.get_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000000000, callbacks=[overfitCallback])



    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000000000, callbacks=[overfitCallback])
    """

def heart():
    df = pd.read_csv('heart.csv', delimiter=',')
    #df.drop('index', inplace=True, axis=1)
    #df.drop('id', inplace=True, axis=1)
    print("-----------")
    opt = SGD(lr=0.01)
    hybridNN = HybridNN(optimizer = 'adam')
    hybridNN.init_model(df, target='output')
    

    labels = hybridNN.df['output']
    features = hybridNN.df.iloc[:, 0:13]

    X = features 

    encoder = LabelEncoder()
    y = np.ravel(labels)
    encoder.fit(y)
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    overfitCallback = EarlyStopping(monitor='loss', patience=100)
    hybridNN.get_model().fit(X_train, y_train, epochs=1000000000, callbacks=[overfitCallback], batch_size = 1)
   
    #get_compiled_model().fit(X_train, y_train, epochs=1000000000, callbacks=[overfitCallback])


def airplane():
    df = pd.read_csv('train.csv', delimiter=',')
    df.drop('index', inplace=True, axis=1)
    df.drop('id', inplace=True, axis=1)
    print("-----------")
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    hybridNN = HybridNN(optimizer = opt)
    df = df.iloc[0:5000, ]
    hybridNN.init_model(df, target='satisfaction')
    
    labels = hybridNN.df['satisfaction']
    features = hybridNN.df.iloc[:, 0:22]

    X = features 

    encoder = LabelEncoder()
    y = np.ravel(labels)
    encoder.fit(y)
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience=100)
    hybridNN.get_model().fit(X_train, y_train, epochs=1000000000, callbacks=[])
    #get_compiled_model().fit(X_train, y_train, epochs=1000000000, callbacks=[overfitCallback])

if __name__ == "__main__":
    #heart()
    #airplane()
    iris()