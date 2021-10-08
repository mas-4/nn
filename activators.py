import numpy as np


def sigmoid(x: np.ndarray, d=False) -> np.ndarray:
    r = 1 / (1 + np.exp(-x))
    if d:
        return r * (1 - r)
    return r


def relu(x: np.ndarray) -> np.ndarray:
    return np.max(0, x)


def softmax(x: np.ndarray, d=True) -> np.ndarray:
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def mse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x - y)**2


def rmse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(mse(x, y))

def onehot(Y, l):
    targets = np.zeros((len(Y), l))
    for y in Y:
        targets[y] = 1.0
    return targets
