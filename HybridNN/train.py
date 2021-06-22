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

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model


def haberman():
    df = pd.read_csv('haberman.csv', delimiter=',')
    print("-----------")
    hybridNN = HybridNN(hidden_activation='sigmoid')
    hybridNN.init_model(df, target='survival_status')
    
    labels = hybridNN.df['survival_status']
    features = hybridNN.df.iloc[:, 0:3]

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
    con_mat = None
    for train, test in kfold.split(X, y):
        print(f'Training for fold {fold_no}')
        """
        hybridNN.reset_model()
        history = hybridNN.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 30,
                         epochs=1000, 
                         #callbacks=[overfitCallback],
                         zero_weight_update=False)
        model = hybridNN.get_model()
        """
        model = get_compiled_model()
        overfitCallback = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 30,
                         epochs=250,
                         callbacks=[])
        
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        if scores[1] * 100 >= max(acc_per_fold):
            best_history = history
            best_model = model
            y_pred = np.argmax(model.predict(X[test]), axis=-1)
            con_mat = tf.math.confusion_matrix(labels = y_not_encoded[test], predictions = y_pred).numpy()
        fold_no = fold_no + 1
        
        
    print(acc_per_fold)
    print(loss_per_fold)
    print(con_mat)

    #con_mat = tf.math.confusion_matrix(labels = y_test_not_encoded, predictions = y_pred).numpy()
    #print(con_mat)

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

def iris():
    df = pd.read_csv('iris.csv', delimiter=',')
    print("-----------")
    hybridNN = HybridNN(hidden_activation='relu')
    hybridNN.init_model(df, target='Species')
    
    labels = hybridNN.df['Species']
    features = hybridNN.df.iloc[:, 0:4]

    X = features.to_numpy()
    
    encoder = LabelEncoder()
    y = np.ravel(labels)
    encoder.fit(y)
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)
    
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    best_model = None
    for train, test in kfold.split(X, y):
        print(f'Training for fold {fold_no}')
        """
        hybridNN.reset_model()
        history = hybridNN.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 13,
                         epochs=250, 
                         #callbacks=[overfitCallback],
                         zero_weight_update=False)
        """
        model = get_compiled_model()
        overfitCallback = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 30,
                         epochs=150,
                         callbacks=[])
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        if scores[1] * 100 >= max(acc_per_fold):
            best_model = history
        fold_no = fold_no + 1
    
        
    print(acc_per_fold)
    print(loss_per_fold)

    plt.plot(best_model.history['accuracy'])
    plt.plot(best_model.history['val_accuracy'])
    plt.title('Best Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(best_model.history['loss'])
    plt.plot(best_model.history['val_loss'])
    plt.title('Best Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    #y_pred = np.argmax(hybridNN.get_model().predict(X_test), axis=-1)
    #con_mat = tf.math.confusion_matrix(labels = y_test_not_encoded, predictions = y_pred).numpy()
    #print(con_mat)
    
 


def heart():
    df = pd.read_csv('heart.csv', delimiter=',')
    print("-----------")
    hybridNN = HybridNN(hidden_activation='sigmoid')
    hybridNN.init_model(df, target='output')
    
    labels = hybridNN.df['output']
    features = hybridNN.df.iloc[:, 0:13]

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
    con_mat = None
    for train, test in kfold.split(X, y):
        print(f'Training for fold {fold_no}')
        """
        hybridNN.reset_model()
        history = hybridNN.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 30,
                         epochs=500, 
                         #callbacks=[overfitCallback],
                         zero_weight_update=False)
        model = hybridNN.get_model()
        """
        model = get_compiled_model()
        overfitCallback = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 30,
                         epochs=250,
                         callbacks=[])
        
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        if scores[1] * 100 >= max(acc_per_fold):
            best_history = history
            best_model = model
            y_pred = np.argmax(model.predict(X[test]), axis=-1)
            con_mat = tf.math.confusion_matrix(labels = y_not_encoded[test], predictions = y_pred).numpy()
        fold_no = fold_no + 1
        
        
    print(acc_per_fold)
    print(loss_per_fold)
    print(con_mat)

    #con_mat = tf.math.confusion_matrix(labels = y_test_not_encoded, predictions = y_pred).numpy()
    #print(con_mat)

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

def student():
    df = pd.read_csv('telescope.csv', delimiter=',')
    df = df.fillna(df.mean())
    print("-----------")
    hybridNN = HybridNN(hidden_activation='relu')
    hybridNN.init_model(df, target='class')
    
    labels = hybridNN.df['class']
    features = hybridNN.df.iloc[:, 0:10]
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
    con_mat = None
    for train, test in kfold.split(X, y):
        print(f'Training for fold {fold_no}')
        """
        hybridNN.reset_model()
        history = hybridNN.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 250,
                         epochs=250, 
                         #callbacks=[overfitCallback],
                         zero_weight_update=True)
        model = hybridNN.get_model()
        """
        model = get_compiled_model()
        overfitCallback = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X[train], y[train], 
                         validation_data=(X[test], y[test]),
                         batch_size = 250,
                         epochs=250,
                         callbacks=[])
        
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        if scores[1] * 100 >= max(acc_per_fold):
            best_history = history
            best_model = model
            y_pred = np.argmax(model.predict(X[test]), axis=-1)
            con_mat = tf.math.confusion_matrix(labels = y_not_encoded[test], predictions = y_pred).numpy()
        fold_no = fold_no + 1
        
        
    print(acc_per_fold)
    print(loss_per_fold)
    print(con_mat)

    #con_mat = tf.math.confusion_matrix(labels = y_test_not_encoded, predictions = y_pred).numpy()
    #print(con_mat)

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
    #student()
    #heart()
    #airplane()
    #iris()
    haberman()
    
    """
    import seaborn as sns
    sns.set(style="white", color_codes=True)

    iris = pd.read_csv("telescope.csv")
    sns.pairplot(iris, hue="class", size=10)    
    plt.show()
    """


    #https://www.kaggle.com/adityakadiwal/water-potability
    #https://www.kaggle.com/fedesoriano/stroke-prediction-dataset