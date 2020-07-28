import torch.nn.functional as F

from bayesnn.layers.diag_prior_module import DiagPriorModule


class BBBConv(DiagPriorModule):
    """
    Convolution layer with a diagonal Normal prior on its weights. To be used with Bayes by BackProp
    (Stochastic Variational Inference). Applying the layer to an input returns a pair: output and KL-term for the loss.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):
        """

        Parameters
        ----------
        in_channels
        out_channels
        bias
        priors
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        shape = (out_channels, in_channels, kernel_size, kernel_size)

        super().__init__(shape, bias=bias, priors=priors)

    def func(self, input, weight, bias):
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
