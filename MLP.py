import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class MLP_stock:
    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        # model.add(tf.keras.layers.BatchNormalization)
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss="mse")
        return (model)

    def train(self, X_train, y_train, bs=32, ntry=10):
        model = self.build_model()
        model.fit(X_train, y_train, batch_size=bs, epochs=100, shuffle=True)
        self.best_model = model
        best_loss = model.evaluate(X_train[-50:], y_train[-50:])

        for i in range(ntry):
            model = self.build_model()
            model.fit(X_train, y_train, batch_size=bs, epochs=100, shuffle=True)
            if model.evaluate(X_train, y_train) < best_loss:
                self.best_model = model
                best_loss = model.evaluate(X_train[-50:], y_train[-50:])

    def predict(self, X_test):
        return (self.best_model.predict(X_test))