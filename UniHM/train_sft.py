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
from UniHM.SFT.utils import *


def _resolve_target_key(dec_key: str, targets: Dict[str, torch.Tensor]) -> Optional[str]:
    aliases = DECODER_KEY_ALIASES.get(dec_key, [dec_key])
    for k in aliases:
        if k in targets:
            return k
    return None

def compute_multihead_l1(model, q_hidden, targets: Dict[str, torch.Tensor], present_robot_keys: List[str], device: torch.device, mano_pose: Optional[torch.Tensor] = None):
    # Map Qwen hidden states back to VQ space per-frame and decode all branches
    mano_vq_hat = model.from_qwen(q_hidden)                  # (B, T, D_vq)
    if mano_vq_hat.dim() != 3:
        raise RuntimeError(f"Unexpected q_hidden->vq_hat shape: {mano_vq_hat.shape}")

    # Reconstruct per-frame sequence via VQVAE decoders
    B = q_hidden.size(0)
    # assume targets have consistent T
    T = targets[next(iter(targets))].size(1) if len(targets) > 0 else mano_vq_hat.size(1)

    vq_dtype = next(model.vqvae.parameters()).dtype
    z_bt = mano_vq_hat.contiguous().view(B * T, model.vq_dim).to(vq_dtype)

    loss = 0.0
    l1 = nn.L1Loss()
    used = 0
    # robot hand decoders
    for i, key in enumerate(present_robot_keys):
        tgt_key = _resolve_target_key(key, targets)
        if tgt_key is None:
            continue
        pred_bt = model.vqvae.decode(z_bt, branch=i).squeeze(-1)  # (B*T, D_out)
        pred = pred_bt.view(B, T, -1)  # (B, T, D_out)
        tgt = targets[tgt_key].to(device)  # (B, T, D_out)
        loss = loss + l1(pred, tgt)
        used += 1
    # MANO decoder (treat as an extra branch)
    if mano_pose is not None and hasattr(model.vqvae, 'mano_decoder') and model.vqvae.mano_decoder is not None:
        pred_bt_mano = model.vqvae.mano_decoder(z_bt).squeeze(-1)  # (B*T, D_mano)
        pred_mano = pred_bt_mano.view(B, T, -1)
        tgt_mano = mano_pose.to(device)
        loss = loss + l1(pred_mano, tgt_mano)
        used += 1
    if used == 0:
        return torch.tensor(0.0, device=device)
    return loss


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vqvae_ckpt = args.vqvae_ckpt
    qwen_id = os.environ.get("QWEN_MODEL", args.qwen_id)

    resume_meta = None
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        resume_meta = torch.load(args.resume_ckpt, map_location="cpu")

    if resume_meta is not None:
        vqvae_kwargs = resume_meta["vqvae_kwargs"]
        qwen_id_for_build = resume_meta.get("qwen_id", qwen_id)
        model = build_qwen_vqvae_aligner(
            vqvae_ckpt_path=vqvae_ckpt,
            vqvae_kwargs=vqvae_kwargs,
            qwen_model_name_or_path=qwen_id_for_build,
            device=device,
            freeze_vqvae=True,
            n_object_tokens=0,
            qwen_dtype=torch.bfloat16,
        )
        present_robot_keys = resume_meta.get("present_robot_keys", [])
    else:
        model, present_robot_keys, vqvae_kwargs = build_model_and_meta(device, args.dataset_path, qwen_id, vqvae_ckpt)

    model.qwen.requires_grad_(args.train_qwen)
    for p in model.vqvae.decoders.parameters():
        p.requires_grad = args.train_decoders
    if hasattr(model.vqvae, 'mano_decoder') and model.vqvae.mano_decoder is not None:
        for p in model.vqvae.mano_decoder.parameters():
            p.requires_grad = args.train_decoders

    # 定义优化器（需在可能恢复其状态之前）
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    # 初始化训练状态变量
    best_val = float('inf')
    best_path = args.save_path
    start_epoch = 1

    if resume_meta is not None:
        model.load_state_dict(resume_meta.get("model_state", {}), strict=False)
        # best_val = resume_meta.get("best_val", best_val)
        start_epoch = 50
        try:
            if "optim_state" in resume_meta:
                optim.load_state_dict(resume_meta["optim_state"])
        except Exception as e:
            tqdm.write(f"Warn: failed to load optimizer state from resume ckpt ({e}); using fresh optimizer.")
        tqdm.write(f"Resumed from {args.resume_ckpt}: start_epoch={start_epoch}, best_val={best_val if best_val != float('inf') else 'inf'}")

    # -------------------- DataLoader build (support file lists) --------------------
    if args.train_list and args.valid_list:
        def _read_list(path):
            with open(path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith('#')]
            return lines
        train_files = _read_list(args.train_list)
        valid_files = _read_list(args.valid_list)
        train_files = [p for p in train_files if os.path.exists(p)]
        valid_files = [p for p in valid_files if os.path.exists(p)]
        if len(train_files) == 0:
            raise RuntimeError(f"No training files found in list {args.train_list}")
        if len(valid_files) == 0:
            raise RuntimeError(f"No validation files found in list {args.valid_list}")
        train_loader, val_loader = build_seq_dataloaders_list(train_files, valid_files, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print("usie glob")
        # train_loader, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=args.batch_size, num_workers=args.num_workers)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        running = 0.0
        for batch in pbar:
            mano_pose = batch["mano_pose"].to(device)            # (B, T, Dm)
            pointcloud = batch["pointcloud"].to(device)          # (B, N, 3)
            objpose = batch["object_pose_seq"].to(device)
            # Respect nopose flag: do not provide object pose to model
            objpose_input = None if args.nopose else objpose
            targets = batch["targets"]
            texts = batch["text"]
            tok = model.tokenizer(texts, padding=True, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}
            
            B, T, S = mano_pose.shape

            if args.mask:
                # Curriculum masking schedule (original behavior)
                if epoch <= 40:
                    mano_mask = torch.ones((B, T), device=device, dtype=mano_pose.dtype)
                elif epoch <= args.epochs * 0.8:
                    progress = (epoch - 40) / (500 * 0.8 - 40)  # 0 -> 1
                    if progress < 0.1:
                        progress = 0.1
                    if progress > 1.0:
                        progress = 1.0
                    keep_ratio = 1.0 - 0.8 * progress  # 1.0 -> 0.2
                    rand = torch.rand((B, T), device=device)
                    mano_mask = (rand < keep_ratio).float()
                    mano_mask[:, :5] = 1.0  # ensure warm frames
                else:
                    mano_mask = torch.zeros((B, T), device=device)
                    mano_mask[:, :5] = 1.0
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                    out = model(
                        mano_pose=mano_pose,
                        object_pointcloud=pointcloud,
                        object_pose_seq=objpose_input,
                        text_inputs=tok,
                        decoder_branch=0,
                        text_position="prefix",
                        mano_mask=mano_mask,
                    )
                    q_hidden = out["qwen_hidden"]
                    loss = compute_multihead_l1(model, q_hidden, targets, present_robot_keys, device, mano_pose=mano_pose)
            else:
                # Two-stage: retargeting (full) + generation (first few frames only)
                full_mask = torch.ones((B, T), device=device, dtype=mano_pose.dtype)
                gen_mask = torch.zeros((B, T), device=device, dtype=mano_pose.dtype); gen_mask[:, :5] = 1.0  # warm start frames
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                    # Retargeting pass
                    out_full = model(
                        mano_pose=mano_pose,
                        object_pointcloud=pointcloud,
                        object_pose_seq=objpose_input,
                        text_inputs=tok,
                        decoder_branch=0,
                        text_position="prefix",
                        mano_mask=full_mask,
                    )
                    l_full = compute_multihead_l1(model, out_full["qwen_hidden"], targets, present_robot_keys, device, mano_pose=mano_pose)
                    # Generation pass (masked future)
                    out_gen = model(
                        mano_pose=mano_pose,
                        object_pointcloud=pointcloud,
                        object_pose_seq=objpose_input,
                        text_inputs=tok,
                        decoder_branch=0,
                        text_position="prefix",
                        mano_mask=gen_mask,
                    )
                    l_gen = compute_multihead_l1(model, out_gen["qwen_hidden"], targets, present_robot_keys, device, mano_pose=mano_pose)
                    loss = 0.5 * (l_full + l_gen)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train = running / max(1, len(train_loader))

        # ---- Validation ----
        model.eval()
        val_running_full = 0.0
        val_running_gen = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [valid]", leave=False)
            for batch in vbar:
                mano_pose = batch["mano_pose"].to(device)
                pointcloud = batch["pointcloud"].to(device)
                objpose = batch["object_pose_seq"].to(device)
                objpose_input = None if args.nopose else objpose
                targets = batch["targets"]
                texts = batch["text"]
                tok = model.tokenizer(texts, padding=True, return_tensors="pt")
                tok = {k: v.to(device) for k, v in tok.items()}
                B, T, _ = mano_pose.shape
                # Retargeting (full)
                out_full = model(
                    mano_pose=mano_pose,
                    object_pointcloud=pointcloud,
                    object_pose_seq=objpose_input,
                    text_inputs=tok,
                    decoder_branch=0,
                    text_position="prefix",
                    mano_mask=torch.ones((B, T), device=device),
                )
                l_full = compute_multihead_l1(model, out_full["qwen_hidden"], targets, present_robot_keys, device, mano_pose=mano_pose)
                val_running_full += float(l_full.item())
                # Generation (only first frame kept)
                first_mask = torch.zeros((B, T), device=device)
                first_mask[:,0] = 1.0
                out_gen = model(
                    mano_pose=mano_pose,
                    object_pointcloud=pointcloud,
                    object_pose_seq=objpose_input,
                    text_inputs=tok,
                    decoder_branch=0,
                    text_position="prefix",
                    mano_mask=first_mask,
                )
                l_gen = compute_multihead_l1(model, out_gen["qwen_hidden"], targets, present_robot_keys, device, mano_pose=mano_pose)
                val_running_gen += float(l_gen.item())
                vbar.set_postfix(retarget=f"{l_full.item():.4f}", gen=f"{l_gen.item():.4f}")

        avg_val_full = val_running_full / max(1, len(val_loader))
        avg_val_gen = val_running_gen / max(1, len(val_loader))

        tqdm.write(f"Epoch {epoch}: train={avg_train:.6f}  val_retarget={avg_val_full:.6f}  val_gen={avg_val_gen:.6f}")

        # Save best based on retargeting loss (could also choose combined metric)
        if avg_val_gen < best_val:
            best_val = avg_val_gen
            ckpt = {
                "model_state": model.state_dict(),
                "vqvae_kwargs": vqvae_kwargs,
                "present_robot_keys": present_robot_keys,
                "qwen_id": qwen_id,
                "optim_state": optim.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }
            torch.save(ckpt, best_path)
            tqdm.write(f"Saved best model to {best_path} (val_gen={best_val:.6f})")
        if epoch % 50 == 0:
            # Periodic save
            ckpt = {
                "model_state": model.state_dict(),
                "vqvae_kwargs": vqvae_kwargs,
                "present_robot_keys": present_robot_keys,
                "qwen_id": qwen_id,
                "optim_state": optim.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }
            torch.save(ckpt, "latest_nomask.pth")
            tqdm.write(f"Saved periodic model to latest.pth (epoch={epoch})")

    tqdm.write("SFT training finished.")


def valid(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.save_path), f"Checkpoint not found: {args.save_path}"
    ckpt = torch.load(args.save_path, map_location="cpu")
    qwen_id = ckpt.get("qwen_id", os.environ.get("QWEN_MODEL", args.qwen_id))
    model = build_qwen_vqvae_aligner(
        vqvae_ckpt_path=args.vqvae_ckpt,
        vqvae_kwargs=ckpt["vqvae_kwargs"],
        qwen_model_name_or_path=qwen_id,
        device=device,
        freeze_vqvae=True,
        n_object_tokens=0,
        qwen_dtype=torch.bfloat16,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    present_robot_keys = ckpt["present_robot_keys"]

    _, val_loader = build_seq_dataloaders(args.seq_glob, batch_size=1, num_workers=max(1, args.num_workers // 4))
    batch = next(iter(val_loader))
    pointcloud = batch["pointcloud"].to(device)
    objpose = batch["object_pose_seq"].to(device)
    objpose_input = None if args.nopose else objpose
    targets = batch["targets"]
    mano_pose = batch["mano_pose"].to(device)
    tok = model.tokenizer(batch["text"], padding=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}

    with torch.no_grad():
        # Validation after training: compute both metrics on a sample
        full_mask = torch.ones((1, objpose.size(1)), device=device)
        first_mask = torch.zeros((1, objpose.size(1)), device=device); first_mask[:,0]=1.0
        out_full = model(mano_pose=None, object_pointcloud=pointcloud, object_pose_seq=objpose_input, text_inputs=tok, decoder_branch=0, gen_mano_len=objpose.size(1))
        out_gen = model(mano_pose=None, object_pointcloud=pointcloud, object_pose_seq=objpose_input, text_inputs=tok, decoder_branch=0, gen_mano_len=objpose.size(1))
        q_hidden_full = out_full["qwen_hidden"]
        q_hidden_gen = out_gen["qwen_hidden"]
        mano_vq_hat_full = model.from_qwen(q_hidden_full)
        mano_vq_hat_gen = model.from_qwen(q_hidden_gen)
        B = mano_vq_hat_full.size(0)
        T = mano_vq_hat_full.size(1)
        vq_dtype = next(model.vqvae.parameters()).dtype
        z_bt_full = mano_vq_hat_full.view(B * T, model.vq_dim).to(vq_dtype)
        z_bt_gen = mano_vq_hat_gen.view(B * T, model.vq_dim).to(vq_dtype)
        preds_full: Dict[str, torch.Tensor] = {}
        preds_gen: Dict[str, torch.Tensor] = {}
        for i, key in enumerate(present_robot_keys):
            pred_bt_full = model.vqvae.decode(z_bt_full, branch=i).squeeze(-1)
            preds_full[key] = pred_bt_full.view(B, T, -1)
            pred_bt_gen = model.vqvae.decode(z_bt_gen, branch=i).squeeze(-1)
            preds_gen[key] = pred_bt_gen.view(B, T, -1)
        # also decode MANO
        if hasattr(model.vqvae, 'mano_decoder') and model.vqvae.mano_decoder is not None:
            preds_full['mano_pose'] = model.vqvae.mano_decoder(z_bt_full).squeeze(-1).view(B, T, -1)
            preds_gen['mano_pose'] = model.vqvae.mano_decoder(z_bt_gen).squeeze(-1).view(B, T, -1)

    for key, pred in preds_full.items():
        tgt_key = _resolve_target_key(key, targets) if key != 'mano_pose' else None
        tgt = targets.get(tgt_key) if tgt_key else (mano_pose if key == 'mano_pose' else None)
        tshape = tuple(tgt.shape) if tgt is not None else None
        print(f"{key} (retargeting): pred shape {tuple(pred.shape)}, target shape {tshape}")
    for key, pred in preds_gen.items():
        tgt_key = _resolve_target_key(key, targets) if key != 'mano_pose' else None
        tgt = targets.get(tgt_key) if tgt_key else (mano_pose if key == 'mano_pose' else None)
        tshape = tuple(tgt.shape) if tgt is not None else None
        print(f"{key} (generation): pred shape {tuple(pred.shape)}, target shape {tshape}")

    print("Validation inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training (generation only) with multi-hand L1 loss, point cloud + object pose conditioning")
    parser.add_argument("--dataset-path", default="/home/main/dex-ICLR/UniHM/UniHM/dataset/dataset.npz")
    parser.add_argument("--seq-glob", default="/home/main/data/robot_data/*.npz")
    parser.add_argument("--vqvae-ckpt", default="/home/main/dex-ICLR/UniHM/UniHM/ckpt/memd/conv1d/memd_conv1d_mano_decoder.pth")
    parser.add_argument("--qwen-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-path", default="sft_best_oak_nomask.pth")
    parser.add_argument("--train-qwen", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--train-decoders", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--mode", choices=["train", "valid"], default="train")
    parser.add_argument("--nopose", action=argparse.BooleanOptionalAction, default=True, help="If set, do not pass object pose sequence to the model (object_pose_seq=None)")
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction, default=True, help="Enable curriculum masking strategy for generation; if false, run two-stage (retargeting + generation) each step.")
    # 新增：文件列表方式
    parser.add_argument("--train-list", type=str, default="train_oak.txt", help="Path to train.txt (one npz path per line)")
    parser.add_argument("--valid-list", type=str, default="valid_oak.txt", help="Path to valid.txt (one npz path per line)")
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
        valid(args)
    else:
        valid(args)