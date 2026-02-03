import torch
import torch.nn as nn
from typing import List, Optional, Union

from .encoder import Encoder, MLPEncoder
from .quantizer import VectorQuantizer
from .decoder import Decoder, MLPDecoder
import time

class MultiDecoderVQVAE(nn.Module):
    """
    VQ-VAE variant with a single encoder and codebook shared across multiple decoders.

    Pipeline (recommended with use_mlp=True):
    - Input x: (B, 1, L) or (B, L)
    - Encoder -> z_e: (B, D) where D == embedding_dim
    - Pre-quantization linear (identity mapping) -> (B, D)
    - VectorQuantizer -> z_q: (B, D)
    - Multiple decoders -> list of reconstructions, each (B, out_dim)

    Args:
        in_dim: Input channel dimension for Conv1d encoder (only used when use_mlp=False).
        h_dim: Hidden channels/width for backbones.
        res_h_dim: Hidden width of residual blocks.
        n_res_layers: Number of residual blocks.
        n_embeddings: Number of codebook entries.
        embedding_dim: Dimension of codebook vectors and bottleneck.
        beta: Commitment cost for VQ loss.
        num_decoders: Number of decoder heads.
        decoder_out_channels: Output dim of each decoder head.
        save_img_embedding_map: Kept for API symmetry (not used here).
        use_mlp: Use MLP-based encoder/decoders working purely with 2D (B, D) tensors.
        input_length: Required when use_mlp=True, length L of the 1D input vector.
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        res_h_dim: int,
        n_res_layers: int,
        n_embeddings: int,
        embedding_dim: int,
        beta: float,
        num_decoders: int = 2,
        input_length: int = 51,
        decoder_out_channels: Optional[List[int]] = None,
        save_img_embedding_map: bool = False,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()

        if num_decoders < 1:
            raise ValueError("num_decoders must be >= 1")

        self.use_mlp = use_mlp
        self.embedding_dim = embedding_dim

        # Encoder produces (B, embedding_dim)
        if self.use_mlp:
            if input_length is None:
                raise ValueError("input_length must be provided when use_mlp=True")
            self.encoder = MLPEncoder(input_length, h_dim, n_res_layers, res_h_dim, embedding_dim=embedding_dim)
        else:
            # Conv1d encoder flattens then projects to (B, embedding_dim)
            self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim, x_shape=input_length, embedding_dim=embedding_dim)

        # Linear keeps size (B, D) -> (B, D); retained for parity/extendability
        self.pre_quantization_conv = nn.Linear(embedding_dim, embedding_dim)

        # Shared codebook on 2D bottlenecks
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)

        # Decoders: for MLP path, each decoder maps (B, D) -> (B, out_dim)
        if self.use_mlp:
            self.decoders = nn.ModuleList(
                [MLPDecoder(embedding_dim, h_dim, n_res_layers, res_h_dim, out_channels=out_ch) for out_ch in decoder_out_channels]
            )
        else:
            # Conv path expects a 3D latent; if used, caller must adapt shapes accordingly.
            self.decoders = nn.ModuleList(
                [Decoder(in_dim, h_dim, n_res_layers, res_h_dim, outdim=out_ch, embedding_dim=embedding_dim) for out_ch in decoder_out_channels]
            )

        # Optional map retained for API symmetry
        self.img_to_embedding_map = {i: [] for i in range(n_embeddings)} if save_img_embedding_map else None

        # Track last batch code indices used (flattened). Useful for utilization stats.
        self.last_code_indices: Optional[torch.Tensor] = None

    # ---- convenience API ----
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to (B, D). Accepts (B, L) or (B, 1, L)."""
        if not self.use_mlp and x.dim() == 2:
            x = x.unsqueeze(1)
        z_e = self.encoder(x)  # (B, D)
        z_e = self.pre_quantization_conv(z_e)  # (B, D)
        return z_e

    def quantize(self, z_e: torch.Tensor):
        return self.vector_quantization(z_e)

    def decode(self, z_q: torch.Tensor, branch: int = 0) -> torch.Tensor:
        if branch < 0 or branch >= len(self.decoders):
            raise IndexError(f"branch index out of range: {branch}")
        return self.decoders[branch](z_q)

    # ---- main forward ----
    def forward(
        self,
        x: torch.Tensor,
        branch: Optional[int] = None,
        return_all: bool = False,
        verbose: bool = False,
    ):
        
        # t0 = time.perf_counter()
        z_e = self.encode(x)  # (B, D)
        # t1 = time.perf_counter()
        embedding_loss, z_q, perplexity, _, min_encoding_indices = self.quantize(z_e)
        # t2 = time.perf_counter()
        # Save indices for utilization statistics (flattened on CPU)
        self.last_code_indices = min_encoding_indices.view(-1).detach().to('cpu')

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("quantized data shape:", z_q.shape)
        # t3 = time.perf_counter()
        if return_all or branch is None:
            # Each decoder returns (B, out_dim, 1) in MLP path; squeeze to (B, out_dim)
            x_hats = [dec(z_q).squeeze(-1) for dec in self.decoders]
            if verbose:
                for i, xh in enumerate(x_hats):
                    print(f"recon[{i}] shape:", xh.shape)
        else:
            x_hats = self.decode(z_q, branch).squeeze(-1)
            if verbose:
                print("recon data shape:", x_hats.shape)
        # t4 = time.perf_counter()    
        # print(f"Timing: encode {t1-t0:.4f}s, quantize {t2-t1:.4f}s, decode {t4-t3:.4f}s")
        return embedding_loss, x_hats, perplexity
