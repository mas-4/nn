import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    r = 1 / (1 + np.exp(-x))
    return r


def sigmoid_d(x: np.ndarray) -> np.ndarray:
    return 1-sigmoid(x)
