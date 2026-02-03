import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResidualStack, MLPResidualStack


class Decoder(nn.Module):
    """
    Conv1d-based decoder (legacy path).
    Input z: (B, C, Lz) or (B, C); returns (B, outdim) after flatten + Linear.
    This path is only used when use_mlp=False in MultiDecoderVQVAE.
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, outdim, embedding_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose1d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose1d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim//2, 1, kernel_size=kernel,
                               stride=stride, padding=1),
            nn.Flatten(start_dim=1),
            nn.Linear(embedding_dim*4, outdim),
        )

    def forward(self, x):
        # Support both (B, C, L=1) and (B, C) bottlenecks; normalize to 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.inverse_conv_stack(x)


class MLPDecoder(nn.Module):
    """
    MLP-based decoder.
    Inputs:
      - x: (B, D) or (B, D, 1), where D == embedding_dim
    Outputs:
      - y: (B, out_channels, 1) -> we typically squeeze(-1) upstream to get (B, out_channels)
    """

    def __init__(self, in_dim: int, h_dim: int, n_res_layers: int, res_h_dim: int, out_channels: int):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, h_dim)
        self.res_stack = MLPResidualStack(dim=h_dim, hidden_dim=res_h_dim, n_layers=n_res_layers)
        # Avoid inplace to prevent autograd versioning errors
        self.act = nn.ReLU(inplace=False)
        self.fc_out = nn.Linear(h_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, D, 1) or (B, D)
        if x.dim() == 3:
            x = x.squeeze(-1)
        h = self.fc_in(x)
        h = self.res_stack(h)
        h = self.act(h)
        y = self.fc_out(h)
        return y.unsqueeze(-1)


if __name__ == "__main__":
    # random data
    x = torch.randn(3, 1, 200)

    # test decoder
    decoder = Decoder(1, 128, 3, 1, 1, embedding_dim=200)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)

    # test MLP decoder with embedding_dim=128
    xm = torch.randn(3, 128)
    mlp_dec = MLPDecoder(128, 128, 3, 64, out_channels=51)
    mlp_out = mlp_dec(xm)
    print('MLP Decoder out shape:', mlp_out.shape)