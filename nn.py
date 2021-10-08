from __future__ import annotations
from typing import Callable
from activators import sigmoid
import tensorflow as tf

import numpy as np


class Network:
    layers = []
    def __init__(self, input_n):
        self.input_n = input_n

    def add_layer(self, Layer_t, neurons, activator):
        if self.layers:
            self.layers.append(
                Layer_t(neurons,
                        activator,
                        self.layers[-1].neurons.shape[0]))
        else:
            self.layers.append(Layer_t(neurons,
                                       activator,
                                       self.input_n))

    def calculate(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output


class LayerDense:
    def __init__(self,
                 neurons: int,
                 activator: Callable,
                 input_n):
        self.neurons = np.random.randn(neurons, input_n)
        self.activator = activator
        self.input_n = input_n

    def forward(self, input):
        return self.activator(np.dot(self.neurons, input))


if __name__ == '__main__':
    input = np.array([
        [2,3,4,5],
        [4,5,6,7],
        [8,9,10,11]
    ])
    nn = Network(784)
    nn.add_layer(LayerDense, 128, sigmoid)
    nn.add_layer(LayerDense, 128, sigmoid)
    nn.add_layer(LayerDense, 2, sigmoid)

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    for img in x_train[:10]:
        print(nn.calculate(img.flatten()))
