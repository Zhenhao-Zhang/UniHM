import os
import argparse
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from glob import glob

from UniHM.SFT import build_qwen_vqvae_aligner
from UniHM.dataset import load_dataset_single, load_dataset_squential

# Order preference for decoders (will auto-filter to present keys from the single-sample metadata)
ROBOT_KEYS_ORDER = [
    "allegro_hand_qpos",
    "shadow_hand_qpos",
    # two variants for SVH
    "svh_hand_qpos",
    "schunk_svh_hand_qpos",
    "leap_hand_qpos",
    "ability_hand_qpos",
    # two variants for Panda
    "panda_hand_qpos",
    "panda_gripper_qpos",
]

# Map decoder canonical keys to acceptable target key aliases in sequential data
DECODER_KEY_ALIASES: Dict[str, List[str]] = {
    "allegro_hand_qpos": ["allegro_hand_qpos"],
    "shadow_hand_qpos": ["shadow_hand_qpos"],
    "svh_hand_qpos": ["svh_hand_qpos", "schunk_svh_hand_qpos"],
    "schunk_svh_hand_qpos": ["svh_hand_qpos", "schunk_svh_hand_qpos"],
    "leap_hand_qpos": ["leap_hand_qpos"],
    "ability_hand_qpos": ["ability_hand_qpos"],
    "panda_hand_qpos": ["panda_hand_qpos", "panda_gripper_qpos"],
    "panda_gripper_qpos": ["panda_hand_qpos", "panda_gripper_qpos"],
}

# Fixed sequence length for batching (crop or pad)
T_FIXED = 70
# Fixed number of points per object for batching point clouds
N_POINTS = 1024


class SeqDataset(Dataset):
    """Dataset of sequential files using load_dataset_squential(file)."""
    def __init__(self, file_glob: str):
        self.files = sorted(glob(file_glob))
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {file_glob}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        result = load_dataset_squential(self.files[idx])
        mano = result["hand_pose"].to(torch.float32)  # (T, Dm)
        pointcloud = result["grasped_obj_point3d"].to(torch.float32)  # (N, 3)
        grasped_with_obj_id = result.get("grasped_with_obj_id", "")
        text = f"grasp object id {grasped_with_obj_id}"  # kept as simple identifier text
        obj_pose_seq = torch.as_tensor(result.get("grasped_obj_pose"), dtype=torch.float32)  # (T, Dp)
        targets: Dict[str, torch.Tensor] = {}
        for k, v in result.items():
            if k.endswith("_qpos"):
                targets[k] = v.to(torch.float32)  # (T, Dk)
        return {"mano_pose": mano, "pointcloud": pointcloud, "object_pose_seq": obj_pose_seq, "text": text, "targets": targets}


def collate_seq(batch: List[Dict[str, any]]):
    manos = []
    objposes = []
    pcls = []
    texts: List[str] = []
    targets_collated: Dict[str, List[torch.Tensor]] = {}

    def sample_pointcloud(pc: torch.Tensor, n_points: int = N_POINTS) -> torch.Tensor:
        # pc: (N, 3)
        N = pc.size(0)
        if N == 0:
            return torch.zeros((n_points, 3), dtype=pc.dtype)
        if N >= n_points:
            idx = torch.randperm(N)[:n_points]
            samp = pc[idx]
        else:
            # Repeat with random picks to reach n_points
            reps = n_points // N
            rem = n_points % N
            pcs = [pc]
            if reps > 1:
                pcs.append(pc.repeat(reps - 1, 1))
            if rem > 0:
                pcs.append(pc[torch.randperm(N)[:rem]])
            samp = torch.cat(pcs, dim=0)
        # Normalize to zero-mean and unit radius
        center = samp.mean(dim=0, keepdim=True)
        samp = samp - center
        scale = samp.norm(dim=1).max().clamp(min=1e-6)
        samp = samp / scale
        return samp

    for b in batch:
        m = b["mano_pose"]  # (T, Dm)
        op = b["object_pose_seq"]  # (T, Dp)
        pc = b["pointcloud"]  # (N, 3)
        T = m.size(0)
        if T >= T_FIXED:
            start = torch.randint(0, T - T_FIXED + 1, (1,)).item()
            m = m[start:start + T_FIXED]
            op = op[start:start + T_FIXED]
            cropped_targets = {k: t[start:start + T_FIXED] for k, t in b["targets"].items()}
        else:
            pad = T_FIXED - T
            # Replicate the last valid frame instead of zero padding (smoother tails)
            if T > 0:
                last_m = m[-1:].expand(pad, -1)
                m = torch.cat([m, last_m], dim=0)
                last_op = op[-1:].expand(pad, -1)
                op = torch.cat([op, last_op], dim=0)
                cropped_targets = {}
                for k, t in b["targets"].items():
                    last_t = t[-1:].expand(pad, -1)
                    cropped_targets[k] = torch.cat([t, last_t], dim=0)
            else:
                # Fallback if an empty sequence appears (shouldn't normally happen)
                m = torch.zeros((T_FIXED, m.size(-1)), dtype=m.dtype)
                op = torch.zeros((T_FIXED, op.size(-1)), dtype=op.dtype)
                cropped_targets = {k: torch.zeros((T_FIXED, t.size(-1)), dtype=t.dtype) for k, t in b["targets"].items()}

        manos.append(m)
        objposes.append(op)
        pcls.append(sample_pointcloud(pc))
        texts.append(b["text"])
        for k, t in cropped_targets.items():
            targets_collated.setdefault(k, []).append(t)

    mano = torch.stack(manos, dim=0)              # (B, T_FIXED, Dm)
    objpose = torch.stack(objposes, dim=0)        # (B, T_FIXED, Dp)
    pcl = torch.stack(pcls, dim=0)                # (B, N_POINTS, 3)
    targets_batch = {k: torch.stack(vs, dim=0) for k, vs in targets_collated.items()}
    return {"mano_pose": mano, "object_pose_seq": objpose, "pointcloud": pcl, "text": texts, "targets": targets_batch}


def build_seq_dataloaders_list(train_list: str, valid_list: str, batch_size: int = 32, num_workers: int = 4):
    class _SliceSeqDataset(SeqDataset):
        def __init__(self, files: List[str]):
            self.files = files
    train_ds = _SliceSeqDataset(train_list)
    val_ds = _SliceSeqDataset(valid_list)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_seq)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 4), collate_fn=collate_seq)
    # 把train_loader和val_loader保存下来
    # torch.save(train_loader, "/home/main/dex-ICLR/UniHM/results/dexycb/train_loader.pth")
    # torch.save(val_loader, "/home/main/dex-ICLR/UniHM/results/dexycb/val_loader.pth")
    return train_loader, val_loader

def build_seq_dataloaders(seq_glob: str, batch_size: int = 32, num_workers: int = 4):
    files = sorted(glob(seq_glob))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {seq_glob}")
    random.shuffle(files)
    split = int(len(files) * 0.8)
    class _SliceSeqDataset(SeqDataset):
        def __init__(self, files: List[str]):
            self.files = files
    train_ds = _SliceSeqDataset(files[:split])
    val_ds = _SliceSeqDataset(files[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_seq)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 4), collate_fn=collate_seq)
    # 把train_loader和val_loader保存下来
    # torch.save(train_loader, "/home/main/dex-ICLR/UniHM/results/dexycb/train_loader.pth")
    # torch.save(val_loader, "/home/main/dex-ICLR/UniHM/results/dexycb/val_loader.pth")
    return train_loader, val_loader


def build_model_and_meta(device: torch.device, single_dataset_path: str, qwen_id: str, vqvae_ckpt: str):
    # Use single-sample metadata to determine decoder outputs and canonical present keys
    single = load_dataset_single(single_dataset_path)
    sample = single[0]
    if hasattr(sample, 'item'):
        sample = sample.item()
    present_robot_keys: List[str] = [k for k in ROBOT_KEYS_ORDER if k in sample]
    out_dims: List[int] = [int(sample[k].shape[0]) for k in present_robot_keys]

    vqvae_kwargs = dict(
        in_dim=1,
        h_dim=128,
        res_h_dim=128,
        n_res_layers=2,
        n_embeddings=8192,
        embedding_dim=512,
        beta=0.25,
        num_decoders=6,
        decoder_out_channels=[22,30,26,22,16,8],
        use_mlp=False,
        input_length=51
    )

    model = build_qwen_vqvae_aligner(
        vqvae_ckpt_path=vqvae_ckpt,
        vqvae_kwargs=vqvae_kwargs,
        qwen_model_name_or_path=qwen_id,
        device=device,
        freeze_vqvae=True,
        n_object_tokens=0,
        qwen_dtype=torch.bfloat16,
    )
    return model, present_robot_keys, vqvae_kwargs


