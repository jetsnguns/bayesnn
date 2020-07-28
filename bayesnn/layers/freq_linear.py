import torch.nn as nn


class FreqLinear(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.lin = nn.Linear()