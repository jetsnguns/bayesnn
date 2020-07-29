import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesnn.layers.bbb_linear import BBBLinear

sigma_w = [4, 3, 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]


class BBBHetRegModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # network with two hidden and one output layer
        self.layer1 = BBBLinear(1, self.hidden_size, priors={'prior_mu': 0,
                                                             'prior_sigma': sigma_w[0] / np.sqrt(1),
                                                             'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                 3. * self.hidden_size)),
                                                             'posterior_rho_initial': (-13., 1e-8)})
        # self.layer2 = BBBLinear(self.hidden_size, self.hidden_size, priors={'prior_mu': 0,
        #                                                                     'prior_sigma': sigma_w[1] / np.sqrt(
        #                                                                         self.hidden_size),
        #                                                                     'posterior_mu_initial': (0, 1. / np.sqrt(
        #                                                                         4. * self.hidden_size)),
        #                                                                     'posterior_rho_initial': (-13., 1e-8)})
        self.layer3 = BBBLinear(self.hidden_size, 2, priors={'prior_mu': 0,
                                                             'prior_sigma': sigma_w[2] / np.sqrt(self.hidden_size),
                                                             'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                 4. * 2.)),
                                                             'posterior_rho_initial': (-13., 1e-8)})

        #self.global_rho = nn.Parameter(torch.Tensor(1))
        #self.global_rho.data.fill_(0.0001)

        self.min_std = 1e-5

    def forward(self, input, sample=True):
        h1, kl_1 = self.layer1(input, sample=sample)
        #h2, kl_2 = self.layer2(nn.functional.relu(h1), sample=sample)
        #h3, kl_3 = self.layer3(nn.functional.relu(h2), sample=sample)
        h3, kl_3 = self.layer3(nn.functional.relu(h1), sample=sample)

        means = h3[:, 0]
        stds = self.min_std + nn.functional.softplus(h3[:, 1]) # + nn.functional.softplus(self.global_rho)
        #kl = kl_1 + kl_2 + kl_3
        kl = kl_1 + kl_3

        return means, stds, kl
