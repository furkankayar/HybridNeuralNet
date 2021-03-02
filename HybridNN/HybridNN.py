import pandas as pd
import numpy as np
from HybridNN_Backend import initialization

TYPE_CATEGORICAL = 0
TYPE_CONTINUOUS = 1

def prepare_dataset(df, target=None):
    types = []
   
    for i, col in zip(df.dtypes, df):
        if i == 'object' or type == 'bool':
            df[col] = df[col].astype('category').cat.codes
            types.append(TYPE_CATEGORICAL)
        else:
            types.append(TYPE_CONTINUOUS)

    return df, np.array(types)


if __name__ == '__main__':
    df = pd.read_csv('iris.csv', delimiter=',')
    df, types = prepare_dataset(df, target='Species')
    arr = df.to_numpy()
    initialization(arr, types)
