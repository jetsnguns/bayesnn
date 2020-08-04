import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesnn.layers.bbb_linear import BBBLinear

sigma_w = [4, 3, 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]


class BBBHetRegModel(nn.Module):
    """
    Bayesian fully connected neural network for heteroscedastic regression following Bayes-By-Backprop algorithm.
    Weights are stochastic, class is designed to work with stochastic variational inference.
    """

    def __init__(self, input_size=1, hidden_size=10, act_func=F.relu):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # network with two hidden and one output layer
        self.layer1 = BBBLinear(input_size, self.hidden_size, priors={'prior_mu': 0,
                                                                      'prior_sigma': sigma_w[0] / np.sqrt(1),
                                                                      'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                          3. * self.hidden_size)),
                                                                      'posterior_rho_initial': (-13., 1e-8)})
        self.layer2 = BBBLinear(self.hidden_size, self.hidden_size, priors={'prior_mu': 0,
                                                                            'prior_sigma': sigma_w[1] / np.sqrt(
                                                                                self.hidden_size),
                                                                            'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                                4. * self.hidden_size)),
                                                                            'posterior_rho_initial': (-13., 1e-8)})
        self.layer3 = BBBLinear(self.hidden_size, 2, priors={'prior_mu': 0,
                                                             'prior_sigma': sigma_w[2] / np.sqrt(self.hidden_size),
                                                             'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                 4. * 2.)),
                                                             'posterior_rho_initial': (-13., 1e-8)})

        self.min_std = 1e-8
        self.act_func = act_func

    def get_kl(self):
        return self.layer1.get_kl() + self.layer2.get_kl() + self.layer3.get_kl()

    def forward(self, input, sample=True):
        h1 = self.layer1(input, sample=sample)
        h2 = self.layer2(self.act_func(h1), sample=sample)
        h3 = self.layer3(self.act_func(h2), sample=sample)

        means = h3[:, :1]
        stds = self.min_std + nn.functional.softplus(h3[:, 1:])

        return means, stds
