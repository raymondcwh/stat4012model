import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
from MLP import MLP_stock
from Data import Data, generate_dataset
from statsmodels.tsa.stattools import acf


tf.random.set_seed(4012)
# STOCKS_TRAIN = ["OXY", "MRO", "GE", "AEP"]
# STOCKS_TEST = ["CL=F"]
STOCKS = ["FB"]
train_len = 900
# train_len = 1500
# download_len = 1800
# df = []

# for stock in STOCKS_TRAIN:
#     df.append(yf.download(stock).iloc[-download_len:]["Adj Close"].iloc[:train_len].values)

for stock in STOCKS:
    df = pd.read_csv(f"./dataset/{stock}.csv", index_col=0, parse_dates=True)
    stock_train = df["Adj Close"].iloc[:train_len].values
    stock_test = df["Adj Close"].iloc[train_len:].values

    X_train, y_train = generate_dataset(stock_train, 5)
    X_test, y_test = generate_dataset(stock_test, 5)


    MLP = MLP_stock()
    MLP.train(X_train, y_train)
    y_pred = np.squeeze(MLP.predict(X_test))

    test_len = len(y_test)

    plt.figure(figsize=(15, 10))
    plt.rcParams["font.size"] = "20"
    plt.plot(range(test_len), y_test, label="true")
    plt.plot(range(test_len), y_pred, label="predict")

    plt.title(f"""{stock} prediction from {df.index[train_len].strftime("%Y-%m-%d")}""")
    plt.ylabel("price", fontsize=15)
    plt.xlabel("trading days", fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    # plt.savefig("Prediction of " + stock + ".png")
    plt.show()

