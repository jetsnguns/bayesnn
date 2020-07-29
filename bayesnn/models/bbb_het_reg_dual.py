import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesnn.layers.bbb_linear import BBBLinear

sigma_w = [4, 3, 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]


class BBBHetRegDualModel(nn.Module):
    def __init__(self, hidden_size, act_func=nn.functional.relu):
        super().__init__()

        self.hidden_size = hidden_size

        # network with two hidden and one output layer
        self.layer1_mu = BBBLinear(1, self.hidden_size, priors={'prior_mu': 0,
                                                                'prior_sigma': sigma_w[0] / np.sqrt(1),
                                                                'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                    3. * self.hidden_size)),
                                                                'posterior_rho_initial': (-13., 1e-8)})
        self.layer2_mu = BBBLinear(self.hidden_size, 1, priors={'prior_mu': 0,
                                                                'prior_sigma': sigma_w[2] / np.sqrt(self.hidden_size),
                                                                'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                    4. * 2.)),
                                                                'posterior_rho_initial': (-13., 1e-8)})

        self.layer1_sigma = BBBLinear(1, self.hidden_size, priors={'prior_mu': 0,
                                                                   'prior_sigma': sigma_w[0] / np.sqrt(1),
                                                                   'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                       3. * self.hidden_size)),
                                                                   'posterior_rho_initial': (-13., 1e-8)})
        self.layer2_sigma = BBBLinear(self.hidden_size, 1, priors={'prior_mu': 0,
                                                                   'prior_sigma': sigma_w[2] / np.sqrt(
                                                                       self.hidden_size),
                                                                   'posterior_mu_initial': (0, 1. / np.sqrt(
                                                                       4. * 2.)),
                                                                   'posterior_rho_initial': (-13., 1e-8)})
        self.act_func = act_func

    def forward(self, input, sample=True):
        h1_mu, kl_1_mu = self.layer1_mu(input, sample=sample)
        h2_mu, kl_2_mu = self.layer2_mu(self.act_func(h1_mu), sample=sample)
        kl_mu = kl_1_mu + kl_2_mu

        mu = h2_mu

        h1_sigma, kl_1_sigma = self.layer1_sigma(input, sample=sample)
        h2_sigma, kl_2_sigma = self.layer2_sigma(self.act_func(h1_sigma), sample=sample)
        kl_sigma = kl_1_sigma + kl_2_sigma

        std = nn.functional.softplus(h2_sigma)

        kl = kl_mu + kl_sigma

        return mu, std, kl
