import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesnn.layers.bbb_linear import BBBLinear


class BBBHetRegModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # network with two hidden and one output layer
        self.layer1 = BBBLinear(1, self.hidden_size, priors={'prior_mu': 0,
                                                             'prior_sigma': 2.5,
                                                             'posterior_mu_initial': (0, 5. / np.sqrt(
                                                                 4. * self.hidden_size)),
                                                             'posterior_rho_initial': (-3., 1.)})
        self.layer2 = BBBLinear(self.hidden_size, self.hidden_size, priors={'prior_mu': 0,
                                                             'prior_sigma': 1.5,
                                                             'posterior_mu_initial': (0, 5. / np.sqrt(
                                                                 4. * self.hidden_size)),
                                                             'posterior_rho_initial': (-3., 1.)})
        self.layer3 = BBBLinear(self.hidden_size, 2, priors={'prior_mu': 0,
                                                             'prior_sigma': 1.25,
                                                             'posterior_mu_initial': (0, 5. / np.sqrt(
                                                                 4. * 2.)),
                                                             'posterior_rho_initial': (-3., 1.)})

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input, sample=True):
        h1, kl_1 = self.layer1(input, sample=sample)
        h2, kl_2 = self.layer2(self.activation(h1), sample=sample)
        h3, kl_3 = self.layer3(self.activation(h2), sample=sample)

        means = h3[:, 0]
        stds = 1e-6 + nn.functional.softplus(h3[:, 1])
        kl = kl_1 + kl_2 + kl_3

        return means, stds, kl
