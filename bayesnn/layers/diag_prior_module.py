import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_kl(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def make_positive(x, eps):
    return F.softplus(x) + eps


class DiagPriorModule(nn.Module):
    def __init__(self, shape, bias=True, priors={}):
        super().__init__()

        priors_def = {
            'prior_mu': 0.,
            'prior_sigma': 0.1,
            'prior_bias_mu': 0.,
            'prior_bias_sigma': 1.,
            'posterior_mu_initial': (0., 0.1),
            'posterior_rho_initial': (-3., 0.1),
        }

        priors_def.update(priors)
        priors = priors_def

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']

        self.prior_bias_mu = priors['prior_bias_mu']
        self.prior_bias_sigma = priors['prior_bias_sigma']

        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.weight_mu = nn.Parameter(torch.empty(shape))
        self.weight_rho = nn.Parameter(torch.empty(shape))

        self.use_bias = bias

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty(shape[0]))
            self.bias_rho = nn.Parameter(torch.empty(shape[0]))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

        self.min_std = 1e-8

    def reset_parameters(self):
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(0., 1e-8)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def kl_weights(self, w_mu, w_sigma):
        return calculate_kl(self.prior_mu, self.prior_sigma, w_mu, w_sigma)

    def kl_bias(self, bias_mu, bias_sigma):
        return calculate_kl(self.prior_bias_mu, self.prior_bias_sigma, bias_mu, bias_sigma)

    def get_weight_sigma(self):
        return make_positive(self.weight_rho, self.min_std)

    def get_bias_sigma(self):
        return make_positive(self.bias_rho, self.min_std)

    def get_kl(self):
        kl = self.kl_weights(self.weight_mu, self.get_weight_sigma())
        if self.use_bias:
            kl += self.kl_bias(self.bias_mu, self.get_bias_sigma())
        return kl

    def func(self, input, weight, bias):
        pass

    def forward(self, input, sample=True):
        if self.training or sample:
            weight = torch.distributions.Normal(self.weight_mu, self.get_weight_sigma()).rsample()

            if self.use_bias:
                bias = torch.distributions.Normal(self.bias_mu, self.get_bias_sigma()).rsample()
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        output = self.func(input, weight, bias)

        return output
