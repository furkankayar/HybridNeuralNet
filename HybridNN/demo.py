import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from HybridNN import HybridNN
import matplotlib.pyplot as plt
import keras.optimizers as optimizers


def iris():
    df = pd.read_csv('iris.csv', delimiter=',')
    df = df.fillna(df.mean())
    hybridNN = HybridNN(hidden_activation='relu')
    hybridNN.init_model(df, target='Species')
    
    labels = hybridNN.df['Species']
    features = hybridNN.df.iloc[:, 0:4]
    X = features.to_numpy()
    
    encoder = LabelEncoder()
    y = np.ravel(labels)
    y_not_encoded = y
    encoder.fit(y)
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)
    
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    best_history = None
    best_model = None
    for train, test in kfold.split(X, y):
        print(f'Training for fold {fold_no}')
        
        hybridNN.reset_model()
        history = hybridNN.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 20,
                         epochs=250, 
                         zero_weight_update=False)
        model = hybridNN.get_model()
  
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        if scores[1] * 100 >= max(acc_per_fold):
            best_history = history
            best_model = model
            y_pred = np.argmax(model.predict(X[test]), axis=-1)
        fold_no = fold_no + 1
        
    print('\nFold\t\tAccuracy\t\tLoss')
    for i in range(5):
        print(f'{i+1}\t\t{acc_per_fold[i]:.2f}\t\t\t{loss_per_fold[i]:.2f}')
    print(f'Avg\t\t{sum(acc_per_fold)/len(acc_per_fold):.2f}\t\t\t{sum(loss_per_fold)/len(loss_per_fold):.2f}')

    plt.plot(best_history.history['accuracy'])
    plt.plot(best_history.history['val_accuracy'])
    plt.title('Best Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(best_history.history['loss'])
    plt.plot(best_history.history['val_loss'])
    plt.title('Best Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



if __name__ == "__main__":
    iris()