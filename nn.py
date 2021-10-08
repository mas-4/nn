from __future__ import annotations
from typing import Callable, Type
from activators import sigmoid, softmax, rmse, onehot
import tensorflow as tf

import numpy as np


class Network:
    layers = []
    def __init__(self, input_n, loss: Callable):
        self.input_n: int = input_n
        self.loss: Callable = loss

    def add_layer(self,
                  Layer_t: Type,
                  neurons: int,
                  activator: Callable):
        if self.layers:
            self.layers.append(
                Layer_t(neurons,
                        activator,
                        self.layers[-1].weights.shape[0]))
        else:
            self.layers.append(Layer_t(neurons,
                                       activator,
                                       self.input_n))

    def predict(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X: np.ndarray, Y: np.ndarray):
        targets = onehot(Y, self.layers[-1].neurons)

        preds = self.predict(X)

        loss = self.loss(preds, targets)

        for layer in self.layers:
            layer.neurons += loss * layer.activator(layer.neurons, d=True)

        totalloss = np.mean(loss, axis=1)
        return np.mean(1 - totalloss)


class LayerDense:
    def __init__(self,
                 neurons: int,
                 activator: Callable,
                 input_n: int):
        self.input_n = input_n # number of weights per neuron
        self.neurons = neurons # number of sets of weights

        self.weights = 0.1 * np.random.randn(neurons, input_n)
        self.biases = np.zeros((neurons, 1))

        self.activator = activator

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.activator(
            np.dot(self.weights, input) + self.biases)


if __name__ == '__main__':

    nn = Network(784, rmse)
    nn.add_layer(LayerDense, 128, sigmoid)
    nn.add_layer(LayerDense, 10, softmax)

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    batch_size = 10

    for i in range(1, len(x_train), batch_size):
        loss = nn.train(x_train[i:i+batch_size].reshape(batch_size,784).T,
                        y_train[i:i+batch_size])
        print(loss)
