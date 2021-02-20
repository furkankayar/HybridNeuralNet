import pandas as pd
import numpy as np
from HybridNN_Backend import initialization
from time import perf_counter


def prepare_dataset(df, target=None):
    df[target] = df[target].astype('category').cat.codes
    return df


if __name__ == '__main__':
    df = pd.read_csv('iris.csv', delimiter=',')
    df = prepare_dataset(df, target='Species')
    arr = df.to_numpy()
    start = perf_counter()
    initialization(arr)
    print('duration: ', perf_counter() - start)