import torch.nn.functional as F

from bayesnn.layers.diag_prior_module import DiagPriorModule


class BBBLinear(DiagPriorModule):
    """
    Linear layer with a diagonal Normal prior on its weights. To be used with Bayes by BackProp
    (Stochastic Variational Inference). Applying the layer to an input returns a pair: output and KL-term for the loss.
    """
    def __init__(self, in_features, out_features, bias=True, priors=None):
        """

        Parameters
        ----------
        in_features
        out_features
        bias
        priors
        """

        self.in_features = in_features
        self.out_features = out_features

        shape = (out_features, in_features)

        super().__init__(shape, bias=bias, priors=priors)

    def func(self, input, weight, bias):
        return F.linear(input, weight, bias)
