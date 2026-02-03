import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResidualStack, MLPResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, x_shape, embedding_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers),
            nn.Flatten(start_dim=1),
            nn.Linear(h_dim * (x_shape // 4), embedding_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x.unsqueeze_(1)  # (B, 1, L)
        # print('Input to encoder shape:', x.shape)
        return self.conv_stack(x)


class MLPEncoder(nn.Module):
    """
    MLP-based encoder for fixed-length vector inputs.
    Accepts x shaped either (B, L) or (B, 1, L). Outputs (B, embedding_dim).
    """

    def __init__(self, in_length: int, hidden_dim: int, n_res_layers: int, res_h_dim: int, embedding_dim: int):
        super().__init__()
        self.in_length = in_length
        self.fc_in = nn.Linear(in_length, hidden_dim)
        self.res_stack = MLPResidualStack(dim=hidden_dim, hidden_dim=res_h_dim, n_layers=n_res_layers)
        self.act = nn.ReLU(inplace=False)
        # project to embedding/bottleneck dimension
        self.fc_out = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, 1, L) or (B, L)
        if x.dim() == 3:
            x = x.squeeze(1)
        h = self.fc_in(x)
        h = self.res_stack(h)
        h = self.act(h)
        # (B, embedding_dim)
        return self.fc_out(h)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 1, 51))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(1, 256, 4, 256)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)

    # x = np.random.random_sample((3, 1, 22))
    # x = torch.tensor(x).float()
    # encoder_out = encoder(x)
    # print('Encoder out shape:', encoder_out.shape)

    # test MLP encoder
    xm = torch.randn(3, 51)
    mlp_enc = MLPEncoder(51, 128, 3, 64, embedding_dim=1024)
    mlp_out = mlp_enc(xm)
    print('MLP Encoder out shape:', mlp_out.shape)