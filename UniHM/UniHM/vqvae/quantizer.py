import torch
import torch.nn as nn
from typing import Optional


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Heatmap of code usage between codebook updates
        # This buffer moves with the module across devices and is saved in state_dict
        self.register_buffer("code_usage_counts", torch.zeros(self.n_e, dtype=torch.long))

    def _flatten_inputs(self, z: torch.Tensor):
        """Flatten inputs to 2D [N, D] while recording shape/permute info for unflattening.
        Supports inputs shaped (B, D) or (B, C, L) with D==e_dim or C==e_dim.
        Returns (z_flattened, restore_fn).
        """
        if z.dim() == 2:
            # (B, D)
            if z.size(1) != self.e_dim:
                raise ValueError(f"Input last dim {z.size(1)} must equal e_dim {self.e_dim}")
            def restore_fn(zq_flat: torch.Tensor):
                return zq_flat  # (B, D)
            return z, restore_fn
        elif z.dim() == 3:
            # (B, C, L) -> (B*L, C) assuming C==e_dim
            B, C, L = z.shape
            if C != self.e_dim:
                # alternatively support (B, L, C)
                if z.size(-1) == self.e_dim:
                    z = z  # (B, C, L=e_dim) not matching; try permute below
                else:
                    raise ValueError(f"For 3D inputs expected channel dim == e_dim ({self.e_dim}), got {C}")
            z_perm = z.permute(0, 2, 1).contiguous()  # (B, L, C)
            z_flat = z_perm.view(-1, self.e_dim)      # (B*L, C)
            def restore_fn(zq_flat: torch.Tensor):
                zq = zq_flat.view(B, L, self.e_dim).permute(0, 2, 1).contiguous()  # (B, C, L)
                return zq
            return z_flat, restore_fn
        else:
            raise ValueError("VectorQuantizer expects input with 2 or 3 dims")

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        Supports z of shape (B, D) or (B, C, L) with D or C == e_dim.
        Returns tensors with the same shape as z.
        """
        # Flatten to (N, D)
        # z_flattened, restore_fn = self._flatten_inputs(z)
        z_flattened = z

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q_flat = torch.matmul(min_encodings, self.embedding.weight)

        # compute loss for embedding
        loss = torch.mean((z_q_flat.detach()-z_flattened)**2) + self.beta * \
            torch.mean((z_q_flat - z_flattened.detach()) ** 2)

        # preserve gradients
        z_q_flat = z_flattened + (z_q_flat - z_flattened).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Update internal heatmap counts (between codebook resets)
        with torch.no_grad():
            idx = min_encoding_indices.view(-1)
            if idx.device != self.code_usage_counts.device:
                idx = idx.to(self.code_usage_counts.device)
            counts = torch.bincount(idx, minlength=self.n_e)
            self.code_usage_counts[:counts.size(0)] += counts.to(self.code_usage_counts.device)

        # Restore to original shape
        # z_q = restore_fn(z_q_flat)
        z_q = z_q_flat


        return loss, z_q, perplexity, min_encodings, min_encoding_indices

    # Utility to manually reset the heatmap if needed
    @torch.no_grad()
    def reset_code_usage_heatmap(self) -> None:
        self.code_usage_counts.zero_()

    # ---- KMeans reinitialization of codebook ----
    @torch.no_grad()
    def kmeans_reset_codebook(
        self,
        data: torch.Tensor,
        num_iters: int = 10,
        max_samples: int = 20000,
        seed: Optional[int] = None,
        verbose: bool = False,
        update_only_cold: bool = True,
        cold_ratio: float = 0.2,
    ) -> None:
        """
        Reinitialize the codebook using K-Means on provided data.

        Args:
            data: Tensor of shape [N, e_dim] containing latent vectors (z_e).
            num_iters: K-Means iterations.
            max_samples: Randomly subsample at most this many points for efficiency.
            seed: Optional random seed for reproducibility.
            verbose: Print simple progress.
            update_only_cold: If True, only update the subset of codebook entries
                that are currently "cold" (low-usage) according to the internal heatmap.
            cold_ratio: Fraction (0,1] of codes with the fewest activations to update
                when update_only_cold is True. Defaults to 0.2 (update bottom 20%).
        """
        if data.dim() != 2 or data.size(1) != self.e_dim:
            raise ValueError(f"data must be of shape [N, {self.e_dim}] but got {tuple(data.shape)}")

        device_cpu = torch.device('cpu')
        x = data.detach().to(device_cpu, dtype=torch.float32)
        N = x.size(0)
        K = self.n_e
        D = self.e_dim

        if N == 0:
            if verbose:
                print("[kmeans_reset_codebook] Empty data, skip.")
            return

        # Subsample for efficiency
        if max_samples is not None and N > max_samples:
            g = torch.Generator(device_cpu)
            if seed is not None:
                g.manual_seed(seed)
            idx = torch.randperm(N, generator=g)[:max_samples]
            x = x[idx]
            N = x.size(0)

        # Initialize centers by random samples
        g = torch.Generator(device_cpu)
        if seed is not None:
            g.manual_seed(seed)
        init_idx = torch.randperm(N, generator=g)[:K]
        centers = x[init_idx].clone()  # [K, D]

        for it in range(num_iters):
            # Assignment step (chunked to save memory)
            labels = torch.empty(N, dtype=torch.long)
            chunk = 4096
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                xx = x[s:e]  # [b, D]
                # distances: (x - c)^2 = x^2 + c^2 - 2 x c^T
                d = (
                    xx.pow(2).sum(1, keepdim=True)
                    + centers.pow(2).sum(1).unsqueeze(0)
                    - 2.0 * xx @ centers.t()
                )  # [b,K]
                labels[s:e] = torch.argmin(d, dim=1)

            # Update step
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(K, dtype=torch.long)
            for k in range(K):
                mask = labels == k
                cnt = int(mask.sum().item())
                if cnt > 0:
                    new_centers[k] = x[mask].mean(dim=0)
                    counts[k] = cnt
                else:
                    # Reinitialize empty cluster to a random point
                    ridx = torch.randint(0, N, (1,), generator=g).item()
                    new_centers[k] = x[ridx]
                    counts[k] = 1

            centers = new_centers
            if verbose:
                n_empty = int((counts == 0).sum().item())
                print(f"[kmeans] iter {it+1}/{num_iters}, empty={n_empty}")

        # Update embedding weights (optionally only for cold codes)
        with torch.no_grad():
            weight_device = self.embedding.weight.device
            weight_dtype = self.embedding.weight.dtype
            if update_only_cold:
                # Determine cold indices based on internal heatmap
                usage_cpu = self.code_usage_counts.detach().to(device_cpu)
                m = int(round(self.n_e * float(cold_ratio)))
                m = max(1, min(self.n_e, m))
                # indices of the m smallest usage counts
                # topk on negative values yields ascending order
                _, cold_indices = torch.topk(-usage_cpu, k=m)

                new_w = self.embedding.weight.data.clone()
                # copy only cold centers
                new_w[cold_indices.to(new_w.device)] = centers[cold_indices].to(new_w.device, dtype=weight_dtype)
                self.embedding.weight.data.copy_(new_w)
            else:
                self.embedding.weight.data.copy_(centers.to(weight_device, dtype=weight_dtype))

            # Reset heatmap after a successful codebook update
            self.code_usage_counts.zero_()