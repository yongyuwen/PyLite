from .imports import *

def flatten(x): return x.view(x.shape[0], -1)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)