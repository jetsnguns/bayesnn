import numpy as np


def true_f(x, e=0):
    return x + 0.3 * np.sin(2 * np.pi * (x + e)) + 0.3 * np.sin(4. * np.pi * (x + e))
    #return np.sin(12.0 * x) + 0.66 * np.cos(25.0 * x) + 3.0
    #return np.sin(x)


def get_data(train_size=100, test_size=100):
    n = train_size + test_size

    np.random.seed(42)

    #X = np.linspace(-2.0, 2.0, num=n)
    X = np.random.uniform(-1.0, 1.0, n)
    e = np.random.normal(0, 0.02, n)
    Y = true_f(X, e=e) + e

    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
