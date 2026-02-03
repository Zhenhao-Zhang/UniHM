import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ..vqvae import MultiDecoderVQVAE
from UniHM.vqvae.decoder import Decoder, MLPDecoder

class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, 9)
        iden = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 9).expand(B, -1)
        x = x + iden
        return x.view(-1, 3, 3)


class STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).expand(B, -1)
        x = x + iden
        return x.view(-1, self.k, self.k)


class PointNetfeat(nn.Module):
    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x: torch.Tensor):
        # x: (B, 3, N)
        n_pts = x.size(2)
        trans = self.stn(x)
        x = x.transpose(2, 1)              # (B, N, 3)
        x = torch.bmm(x, trans)            # (B, N, 3)
        x = x.transpose(2, 1).contiguous() # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1).contiguous()
        else:
            trans_feat = None

        pointfeat = x  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.bn3(self.conv3(x))          # (B, 1024, N)
        x = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x_exp = x.view(-1, 1024, 1).expand(-1, -1, n_pts)
            return torch.cat([x_exp, pointfeat], dim=1), trans, trans_feat


class SimplePointNet(nn.Module):
    """
    Stronger PointNet encoder: (B, N, 3) -> global 1024-d -> project to (B, 1, H)
    """
    def __init__(self, out_dim: int, feature_transform: bool = False):
        super().__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.proj = nn.Linear(1024, out_dim)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: (B, N, 3)
        assert pc.dim() == 3 and pc.size(-1) == 3, f"Expected (B,N,3) point cloud, got {tuple(pc.shape)}"
        x = pc.transpose(1, 2).contiguous()  # (B, 3, N)
        glob, _, _ = self.feat(x)            # (B, 1024)
        h = self.proj(glob)                  # (B, H)
        return h.unsqueeze(1)                # (B, 1, H)


class QwenVQVAEAligner(nn.Module):
    """
    Bridge between your VQVAE (encoder/decoder) and a Qwen model (e.g., 'qwen3-0.6b').
    - Encodes MANO pose -> quantized tokens (B, T, C=embedding_dim)
    - Projects tokens to Qwen hidden size and feeds as inputs_embeds
    - Projects Qwen outputs back to VQ dim and decodes per-frame via VQVAE decoders

    Modalities:
      - MANO pose sequences
      - Object point cloud (B, N, 3) encoded via a lightweight PointNet into 1 token
      - Text tokens via tokenizer
    """
    def __init__(
        self,
        vqvae: MultiDecoderVQVAE,
        qwen_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        freeze_vqvae: bool = True,
        qwen_dtype: Optional[torch.dtype] = torch.float,
        add_modality_embeddings: bool = True,
        n_object_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.vqvae = vqvae
        self.vqvae.eval()
        self.freeze_vqvae = freeze_vqvae
        if freeze_vqvae:
            for p in self.vqvae.parameters():
                p.requires_grad = False

        # Load tokenizer first so we can extend special tokens, then load Qwen
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path, trust_remote_code=True)
        # Special tokens: MANO, point cloud, object pose sequence
        # <point> ... </point> now wrap point cloud single token
        # <obj> ... </obj> wrap grasped object pose sequence tokens
        self.special_tokens = ["<mano>", "</mano>", "<point>", "</point>", "<obj>", "</obj>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        self.special_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens)

        cfg = AutoConfig.from_pretrained(qwen_model_name_or_path, trust_remote_code=True)
        if qwen_dtype is not None:
            cfg.torch_dtype = qwen_dtype
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            config=cfg,
            torch_dtype=qwen_dtype,
            trust_remote_code=True,
        )
        self.qwen.resize_token_embeddings(len(self.tokenizer))
        self.qwen.requires_grad_(True)
        self.hidden_size = self.qwen.config.hidden_size
        self.model_dtype = next(self.qwen.parameters()).dtype

        self.vq_dim = self._infer_vq_dim()
        self.to_qwen = nn.Linear(self.vq_dim, self.hidden_size).to(self.model_dtype)
        self.from_qwen = nn.Linear(self.hidden_size, self.vq_dim).to(self.model_dtype)

        # Encoders
        self.pc_encoder = SimplePointNet(out_dim=self.hidden_size)  # point cloud -> 1 token
        # Object pose projection (lazy init if dim not known at build time)
        self.obj_pose_proj: Optional[nn.Linear] = None

        self.mano_blank_vq = nn.Parameter(torch.zeros(1, self.vq_dim, dtype=self.model_dtype))
        nn.init.trunc_normal_(self.mano_blank_vq, std=0.02)
        self.point_blank_h = nn.Parameter(torch.zeros(1, self.hidden_size, dtype=self.model_dtype))
        nn.init.trunc_normal_(self.point_blank_h, std=0.02)
        self.obj_blank_h = nn.Parameter(torch.zeros(1, self.hidden_size, dtype=self.model_dtype))
        nn.init.trunc_normal_(self.obj_blank_h, std=0.02)

        self.add_modality_embeddings = add_modality_embeddings
        self.n_object_tokens = 0
        self._obj_inited = True

    def _infer_vq_dim(self) -> int:
        # Use vector_quantization embedding_dim as the bottleneck channel
        return getattr(self.vqvae.vector_quantization, "embedding_dim", None) or 512

    @torch.no_grad()
    def _encode_mano_with_vqvae(
        self, mano_pose: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_q_tokens: quantized tokens per frame, shape (B, T, C_vq)
          z_q_bt: flattened per-frame tokens for decoding, shape (B*T, C_vq)
        """
        x = mano_pose
        if x.dim() == 3:
            # (B, T, D) -> (B*T, D)
            B, T, Dm = x.shape
            x_bt = x.reshape(B * T, Dm)
        elif x.dim() == 2:
            # (B, D) -> single frame per batch
            B, Dm = x.shape
            T = 1
            x_bt = x
        elif x.dim() == 1:
            # (D,) -> (1, D)
            Dm = x.size(0)
            B, T = 1, 1
            x_bt = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected mano_pose dims: {tuple(x.shape)}")

        # Encode to (B*T, D) then quantize to (B*T, D)
        z_e = self.vqvae.encode(x_bt)
        _, z_q, _, _, _ = self.vqvae.quantize(z_e)
        z_q_tokens = z_q.view(B, T, -1).contiguous()  # (B, T, D_vq)
        return z_q_tokens, z_q

    def _embed_special(self, token_str: str, batch: int, device: torch.device) -> torch.Tensor:
        """Lookup and return embedding for a single special token, expanded to batch.
        Shape: (B, 1, H)
        """
        tok_id = self.tokenizer.convert_tokens_to_ids(token_str)
        tok_emb = self.qwen.get_input_embeddings()
        ids = torch.full((batch, 1), tok_id, dtype=torch.long, device=device)
        return tok_emb(ids).to(self.model_dtype)

    def _build_inputs_embeds(
        self,
        mano_embeds: Optional[torch.Tensor],   # (B, Tm, H)
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        point_embeds: Optional[torch.Tensor] = None,  # (B, 1, H)
        objpose_embeds: Optional[torch.Tensor] = None,  # (B, To, H)
        text_position: str = "prefix",
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[slice, slice, Tuple[int, int]]]:
        """
        Returns:
          inputs_embeds: (B, L_total, H)
          attention_mask: (B, L_total)
          slices: (text_slice, obj_slice, (mano_start, mano_stop))
        """
        # Modified: point cloud wrapped by <point></point>, object pose sequence by <obj></obj>
        assert (mano_embeds is not None) and mano_embeds.dim() == 3
        B, Tm, _ = mano_embeds.shape
        device = mano_embeds.device
        parts: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        def append_text():
            text_len_local = 0
            if text_inputs is not None and "input_ids" in text_inputs:
                input_ids = text_inputs["input_ids"].to(device)
                text_mask = text_inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
                tok_emb = self.qwen.get_input_embeddings()
                te = tok_emb(input_ids).to(self.model_dtype)
                parts.append(te)
                masks.append(text_mask)
                text_len_local = te.size(1)
            return text_len_local

        # Assemble according to text position
        text_len = 0
        if text_position == "prefix":
            text_len = append_text()

        # <point> single token </point>
        if point_embeds is not None:
            parts.append(self._embed_special("<point>", B, device))
            masks.append(torch.ones((B,1), dtype=torch.long, device=device))
            parts.append(point_embeds)
            masks.append(torch.ones((B, point_embeds.size(1)), dtype=torch.long, device=device))
            parts.append(self._embed_special("</point>", B, device))
            masks.append(torch.ones((B,1), dtype=torch.long, device=device))

        # <obj> object pose sequence </obj>
        obj_len = 0
        if objpose_embeds is not None:
            obj_len = objpose_embeds.size(1)
            parts.append(self._embed_special("<obj>", B, device))
            masks.append(torch.ones((B,1), dtype=torch.long, device=device))
            parts.append(objpose_embeds)
            masks.append(torch.ones((B, obj_len), dtype=torch.long, device=device))
            parts.append(self._embed_special("</obj>", B, device))
            masks.append(torch.ones((B,1), dtype=torch.long, device=device))

        # <mano> sequence </mano>
        parts.append(self._embed_special("<mano>", B, device))
        masks.append(torch.ones((B,1), dtype=torch.long, device=device))
        parts.append(mano_embeds)
        masks.append(torch.ones((B, Tm), dtype=torch.long, device=device))
        parts.append(self._embed_special("</mano>", B, device))
        masks.append(torch.ones((B,1), dtype=torch.long, device=device))

        if text_position == "suffix":
            text_len = append_text()

        # Concat
        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.cat(masks, dim=1)

        # Compute slices
        cursor = 0
        text_slice = slice(0,0)
        if text_position == "prefix":
            text_slice = slice(cursor, cursor + text_len)
            cursor += text_len
        # point tokens (skip slice tracking)
        if point_embeds is not None:
            cursor += 1 + point_embeds.size(1) + 1
        # object pose slice returned as obj_slice
        if objpose_embeds is not None:
            cursor += 1  # <obj>
            obj_slice = slice(cursor, cursor + obj_len)
            cursor += obj_len
            cursor += 1  # </obj>
        else:
            obj_slice = slice(cursor, cursor)
        cursor += 1  # <mano>
        mano_start = cursor
        cursor += Tm
        mano_stop = cursor
        cursor += 1  # </mano>
        if text_position == "suffix":
            text_slice = slice(cursor, cursor + text_len)
            cursor += text_len
        return inputs_embeds, attention_mask, (text_slice, obj_slice, (mano_start, mano_stop))

    def forward(
        self,
        mano_pose: Optional[torch.Tensor],
        object_pointcloud: Optional[torch.Tensor] = None,
        object_pose_seq: Optional[torch.Tensor] = None,  # (B, T, Dp)
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        decoder_branch: int = 0,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        gen_mano_len: Optional[int] = None,
        text_position: str = "prefix",
        mano_mask: Optional[torch.Tensor] = None,  # (B,T) 1=keep encoded token, 0=blank
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            'reconstruction': decoded output from VQVAE decoder per-frame (B, T, D_out),
            'mano_tokens': (B, T, C_vq),
            'qwen_hidden': (B, T, H),
            'extras': {...}
          }
        """
        # Encode MANO
        if mano_pose is not None:
            with torch.set_grad_enabled(not self.freeze_vqvae):
                mano_tokens, _ = self._encode_mano_with_vqvae(mano_pose)
        else:
            assert gen_mano_len is not None and gen_mano_len > 0
            B = (text_inputs["input_ids"].size(0) if text_inputs is not None else 1)
            mano_tokens = self.mano_blank_vq.expand(B, gen_mano_len, -1)
        # Apply optional mask: replace masked frames with blank token embedding pre-projection
        if mano_mask is not None:
            assert mano_mask.shape[:2] == mano_tokens.shape[:2], f"mano_mask shape {mano_mask.shape} mismatch tokens {mano_tokens.shape}";
            # broadcast blank token
            blank = self.mano_blank_vq.to(mano_tokens.dtype)
            # mano_mask: (B,T) -> (B,T,1)
            mm = mano_mask.to(mano_tokens.device).unsqueeze(-1)
            mano_tokens = mano_tokens * mm + (1 - mm) * blank
        mano_embeds = self.to_qwen(mano_tokens.to(self.model_dtype))

        # Point cloud -> 1 token
        point_embeds = None
        if object_pointcloud is not None:
            point_embeds = self.pc_encoder(object_pointcloud.to(torch.float32)).to(self.model_dtype)
        # Object pose sequence -> tokens
        objpose_embeds = None
        if object_pose_seq is not None:
            assert object_pose_seq.dim() == 3, "object_pose_seq must be (B,T,Dp)"
            B, To, Dp = object_pose_seq.shape
            # Lazy init or relocate projection layer to match device + dtype
            target_device = object_pose_seq.device
            if self.obj_pose_proj is None:
                self.obj_pose_proj = nn.Linear(Dp, self.hidden_size)
            # Move to correct device / dtype if needed
            if self.obj_pose_proj.weight.device != target_device:
                self.obj_pose_proj = self.obj_pose_proj.to(target_device)
            if self.obj_pose_proj.weight.dtype != self.model_dtype:
                self.obj_pose_proj = self.obj_pose_proj.to(self.model_dtype)
            objpose_embeds = self.obj_pose_proj(object_pose_seq.to(dtype=self.model_dtype))  # (B,To,H)
        inputs_embeds, attention_mask, slices = self._build_inputs_embeds(
            mano_embeds=mano_embeds,
            text_inputs=text_inputs,
            point_embeds=point_embeds,
            objpose_embeds=objpose_embeds,
            text_position=text_position,
        )
        text_slice, obj_slice, (mano_start, mano_stop) = slices

        # Run Qwen (reasoning removed: no language modeling loss)
        out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
            labels=None,  # no LM loss
        )
        last_hidden = out.hidden_states[-1]
        mano_h = last_hidden[:, mano_start:mano_stop, :]
        mano_vq_hat = self.from_qwen(mano_h)

        vq_dtype = next(self.vqvae.parameters()).dtype
        if mano_pose is not None and mano_pose.dim() == 3:
            B, T, _ = mano_pose.shape
        elif mano_pose is not None and mano_pose.dim() == 2:
            B, T = mano_pose.size(0), 1
        else:
            B = last_hidden.size(0)
            T = mano_h.size(1)
        z_bt = mano_vq_hat.contiguous().view(B * T, self.vq_dim).to(vq_dtype)
        recon_bt = self.vqvae.decode(z_bt, branch=decoder_branch)
        recon = recon_bt.squeeze(-1).view(B, T, -1)

        return {
            "reconstruction": recon,
            "mano_tokens": mano_tokens,
            "qwen_hidden": mano_h,
            "extras": {
                "attention_mask": attention_mask,
                "text_slice": (text_slice.start, text_slice.stop),
                "mano_slice": (mano_start, mano_stop),
                "tokens_per_frame": 1,
                "T": T,
                "lm_loss": None,
            },
        }

    @torch.no_grad()
    def _get_codebook(self) -> torch.Tensor:
        """Try to fetch VQ codebook weight tensor (K, C_vq)."""
        vq = self.vqvae.vector_quantization
        cand = None
        for attr in ["embedding", "codebook", "embed", "embeddings"]:
            if hasattr(vq, attr):
                obj = getattr(vq, attr)
                if isinstance(obj, nn.Embedding):
                    cand = obj.weight
                elif isinstance(obj, torch.Tensor):
                    cand = obj
                elif hasattr(obj, "weight"):
                    cand = obj.weight
            if cand is not None:
                break
        if cand is None:
            raise RuntimeError("Could not locate codebook weights in vector_quantization module.")
        return cand  # (K, C_vq)

    @torch.no_grad()
    def _sample_code(self, pred_vq: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given predicted continuous VQ embedding (B, C_vq), pick a codebook entry.
        Returns (indices, embeddings) where embeddings shape (B, C_vq)."""
        codebook = self._get_codebook()  # (K, C)
        B, C = pred_vq.shape
        K = codebook.size(0)
        # Squared distance -> logits
        # d2 = ||pred - e||^2 = pred^2 + e^2 - 2 pred·e ; constants relative across codes can be ignored
        # Use cosine-ish by just doing negative L2
        d2 = (pred_vq.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)  # (B,K)
        logits = -d2  # higher is better
        if temperature is not None and temperature > 0:
            logits = logits / temperature
        # Top-k filter
        if top_k and top_k > 0 and top_k < K:
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            logits = mask
        # Top-p (nucleus) filter
        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            cut_mask = cum > top_p
            # Ensure at least 1 token kept
            cut_mask[..., 0] = False
            sorted_logits[cut_mask] = float('-inf')
            # Scatter back
            new_logits = torch.full_like(logits, float('-inf'))
            new_logits.scatter_(1, sorted_idx, sorted_logits)
            logits = new_logits
        probs = torch.softmax(logits, dim=-1)
        # If deterministic (all mass on one or sampling disabled), argmax path
        if (top_k == 1) or (top_k == 0 and top_p >= 1.0 and (temperature == 0 or temperature is None)):
            indices = torch.argmax(probs, dim=-1)
        else:
            indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        emb = codebook[indices]
        return indices, emb

    @torch.no_grad()
    def generate_mano_autoreg(
        self,
        gen_mano_len: int,
        decoder_branch: int = 0,
        texts: Optional[List[str]] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        object_pointcloud: Optional[torch.Tensor] = None,
        object_pose_seq: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        text_position: str = "prefix",
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Autoregressively generate MANO VQ tokens using next-token prediction on hidden states.
        Minimal intrusion: reuse projection + _build_inputs_embeds; loop by appending a blank slot.
        """
        assert gen_mano_len > 0
        if device is None:
            device = next(self.parameters()).device
        # Prepare text inputs
        if text_inputs is None and texts is not None:
            text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        B = 1
        if text_inputs is not None:
            B = text_inputs["input_ids"].size(0)
        elif object_pointcloud is not None:
            B = object_pointcloud.size(0)
        elif object_pose_seq is not None:
            B = object_pose_seq.size(0)
        # Prepare auxiliary modalities (static across steps)
        point_embeds = None
        if object_pointcloud is not None:
            point_embeds = self.pc_encoder(object_pointcloud.to(device, dtype=torch.float32)).to(self.model_dtype)
        objpose_embeds = None
        if object_pose_seq is not None:
            assert object_pose_seq.dim() == 3
            B2, To, Dp = object_pose_seq.shape
            assert B2 == B
            if self.obj_pose_proj is None:
                self.obj_pose_proj = nn.Linear(Dp, self.hidden_size).to(device).to(self.model_dtype)
            elif self.obj_pose_proj.weight.device != device:
                self.obj_pose_proj = self.obj_pose_proj.to(device)
            if self.obj_pose_proj.weight.dtype != self.model_dtype:
                self.obj_pose_proj = self.obj_pose_proj.to(self.model_dtype)
            objpose_embeds = self.obj_pose_proj(object_pose_seq.to(device=device, dtype=self.model_dtype))  # (B,To,H)
        # Storage
        gen_tokens: List[torch.Tensor] = []  # list of (B, C_vq)
        gen_indices: List[torch.Tensor] = []
        blank_token = self.mano_blank_vq.to(device)  # (1, C_vq)
        for step in range(gen_mano_len):
            if step == 0:
                # Seed with single blank inside <mano></mano>
                current_vq = blank_token.expand(B, 1, -1)  # (B,1,C_vq)
            else:
                prev = torch.stack(gen_tokens, dim=1)  # (B, step, C_vq)
                current_vq = torch.cat([prev, blank_token.expand(B, 1, -1)], dim=1)  # add placeholder
            mano_embeds = self.to_qwen(current_vq.to(self.model_dtype))  # (B, Lm, H)
            inputs_embeds, attention_mask, slices = self._build_inputs_embeds(
                mano_embeds=mano_embeds,
                text_inputs=text_inputs,
                point_embeds=point_embeds,
                objpose_embeds=objpose_embeds,
                text_position=text_position,
            )
            _, _, (mano_start, mano_stop) = slices
            out = self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                labels=None,
            )
            last_hidden = out.hidden_states[-1]
            mano_h = last_hidden[:, mano_start:mano_stop, :]  # (B, current_len, H)
            # Predict embedding for the last (placeholder) position
            pred_hidden = mano_h[:, -1, :]
            pred_vq = self.from_qwen(pred_hidden)  # (B, C_vq)
            # Map to discrete code via sampling
            idx, emb = self._sample_code(pred_vq.to(self.mano_blank_vq.dtype), temperature=temperature, top_k=top_k, top_p=top_p)
            gen_indices.append(idx)
            gen_tokens.append(emb)  # emb already (B, C_vq)
        # Stack final tokens
        final_tokens = torch.stack(gen_tokens, dim=1)  # (B, T, C_vq)
        # Decode
        vq_dtype = next(self.vqvae.parameters()).dtype
        z_bt = final_tokens.view(B * gen_mano_len, -1).to(vq_dtype)
        recon_bt = self.vqvae.decode(z_bt, branch=decoder_branch)
        recon = recon_bt.squeeze(-1).view(B, gen_mano_len, -1)
        return {
            "reconstruction": recon,
            "mano_tokens": final_tokens,  # continuous embeddings of chosen codes
            "indices": torch.stack(gen_indices, dim=1),  # (B, T)
            "extras": {
                "T": gen_mano_len,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
        }

    @torch.no_grad()
    def infer_poses_from_text(
        self,
        texts: List[str],
        decoder_branch: int,
        gen_mano_len: int,
        object_pointcloud: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Given text (+ optional point cloud), generate MANO sequence via blank MANO tokens."""
        if device is None:
            device = next(self.parameters()).device
        tok = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        out = self.forward(
            mano_pose=None,
            object_pointcloud=object_pointcloud.to(device) if object_pointcloud is not None else None,
            text_inputs=tok,
            decoder_branch=decoder_branch,
            gen_mano_len=gen_mano_len,
            text_position="prefix",
            use_cache=False,
        )
        return {"mano_recon": out["reconstruction"]}


def build_qwen_vqvae_aligner(
    vqvae_ckpt_path: Optional[str],
    vqvae_kwargs: Dict[str, Any],
    qwen_model_name_or_path: str = "Qwen/Qwen3-0.6B",
    device: Optional[Union[str, torch.device]] = None,
    freeze_vqvae: bool = True,
    n_object_tokens: int = 0,
    qwen_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> QwenVQVAEAligner:
    """
    Helper to instantiate VQVAE + load weights, then wrap with QwenVQVAEAligner.
    vqvae_kwargs should match how you trained your VQVAE (dims, layers, etc.).
    """
    vqvae = MultiDecoderVQVAE(**vqvae_kwargs)
    if vqvae_kwargs.get("use_mlp", False):
        mano_dec = MLPDecoder(vqvae_kwargs.get("embedding_dim", 512),
                              vqvae_kwargs.get("h_dim", 128),
                              vqvae_kwargs.get("n_res_layers", 2),
                              vqvae_kwargs.get("res_h_dim", 128),
                              out_channels=51)
    else:
        mano_dec = Decoder(vqvae_kwargs.get("in_dim", 1),
                           vqvae_kwargs.get("h_dim", 128),
                           vqvae_kwargs.get("n_res_layers", 2),
                           vqvae_kwargs.get("res_h_dim", 128),
                           outdim=51,
                           embedding_dim=vqvae_kwargs.get("embedding_dim", 512))
    vqvae.mano_decoder = mano_dec.to(device)
    if vqvae_ckpt_path:
        sd = torch.load(vqvae_ckpt_path, map_location="cpu")
        vqvae.load_state_dict(sd, strict=False)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = vqvae.to(device)

    aligner = QwenVQVAEAligner(
        vqvae=vqvae,
        qwen_model_name_or_path=qwen_model_name_or_path,
        freeze_vqvae=freeze_vqvae,
        n_object_tokens=n_object_tokens,
        qwen_dtype=qwen_dtype,
    ).to(device)

    return aligner