import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        # create distinct residual layers (avoid shared weights)
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


# ---- MLP residual variants ----
class MLPResidualLayer(nn.Module):
    """
    Residual block for MLP pathway operating on 2D tensors (B, H).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        # Avoid inplace to prevent autograd versioning errors
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = residual + out
        return out


class MLPResidualStack(nn.Module):
    """
    A stack of MLPResidualLayer blocks.
    """

    def __init__(self, dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([MLPResidualLayer(dim, hidden_dim) for _ in range(n_layers)])
        # Avoid inplace to prevent autograd versioning errors
        self.out_act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out_act(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 200))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)

    # test MLP residual stack
    xm = torch.randn(3, 128)
    mlp_stack = MLPResidualStack(128, 64, 3)
    xm_out = mlp_stack(xm)
    print('MLP Res Stack out shape:', xm_out.shape)