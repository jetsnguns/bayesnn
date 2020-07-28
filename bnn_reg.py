import numpy as np
import torch


from bayesnn.misc import ModuleWrapper, FlattenLayer
from bayesnn.layers.BBBLinear import BBBLinear


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BBBFC3(ModuleWrapper):
    def __init__(self, inputs=1, size=10):
        super().__init__()

        self.num_inputs = inputs
        self.size = size

        self.fc1 = torch.nn.Linear(in_features=self.num_inputs, out_features=self.size)
        self.act1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(in_features=self.size, out_features=self.size)
        self.act2 = torch.nn.ReLU()

        self.flatten = FlattenLayer(num_features=self.size)

        self.fc3 = BBBLinear(in_features=self.size, out_features=1)
        self.act3 = torch.nn.ReLU()


def true_f(x):
    return np.sin(12.0 * x) + 0.66 * np.cos(25.0 * x) + 3.0


def get_data(train_size=100, test_size=100):
    n = train_size + test_size

    np.random.seed(42)

    #X = np.linspace(-2.0, 2.0, num=n)
    X = np.random.uniform(-2.0, 2.0, n)
    Y = true_f(X) + np.random.normal(0, 0.1, n)

    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


if __name__ == "__main__":
