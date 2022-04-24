import pandas as pd
import numpy as np

class Data:
    def __init__(self, type):
        self.df = pd.read_csv("./dataset/dataset.csv", header=[0,1], index_col=0, parse_dates=True)
        self.type = type

        return



def generate_dataset(price, seq_len):
    X_list, y_list = [], []
    for i in range(len(price) - seq_len):
        X = np.array(price[i:i + seq_len])
        y = np.array([price[i + seq_len]])
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list), np.array(y_list)