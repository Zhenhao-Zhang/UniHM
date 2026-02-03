from UniHM.dataset import HandDataset, load_dataset_single
from UniHM.vqvae import MultiDecoderVQVAE
from UniHM.vqvae.encoder import Encoder, MLPEncoder
import torch
import random
import json
import os
import argparse
from typing import List, Optional

ROBOT_KEYS_ORDER = [
    "allegro_hand_qpos",
    "shadow_hand_qpos",
    "svh_hand_qpos",
    "leap_hand_qpos",
    "ability_hand_qpos",
    "panda_hand_qpos",
]

# Will be initialized from args in main
device = torch.device("cpu")


def compute_loss(ypred, ydict):
    """Sum L1 over present robot targets in fixed order."""
    def loss_func(x, y):
        return torch.nn.functional.l1_loss(x, y)

    ylist = []
    for k in ROBOT_KEYS_ORDER:
        if k in ydict:
            ylist.append(ydict[k])
    losses = [loss_func(x, y.to(device)) for x, y in zip(ypred, ylist)]
    return sum(losses)


# ---- New: Training phase helpers ----

def phase1_distill(
    student: torch.nn.Module,
    teacher_encoder: torch.nn.Module,
    train_dataloader,
    valid_dataloader,
    device: torch.device,
    save_path: str,
    epochs: int,
    lr: float,
    wd: float,
    rkey: str,
    r_in_len: int,
):
    """Phase 1: Distill student encoder to match teacher encoder outputs."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
    best_val = float('inf')

    print(f"\n==== [Phase 1] Distill student encoder for {rkey} (input_len={r_in_len}) ====")
    for epoch in range(epochs):
        # Train
        student.train()
        train_loss = 0.0
        train_batches = 0
        for x_mano, ydict in train_dataloader:
            if rkey not in ydict:
                continue
            x_mano = x_mano.to(device)
            x_robot = ydict[rkey].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z_teacher = teacher_encoder(x_mano)  # (B, D)
            z_student = student(x_robot)            # (B, D)

            loss = torch.nn.functional.l1_loss(z_student, z_teacher)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Valid
        student.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for x_mano, ydict in valid_dataloader:
                if rkey not in ydict:
                    continue
                x_mano = x_mano.to(device)
                x_robot = ydict[rkey].to(device)

                z_teacher = teacher_encoder(x_mano)
                z_student = student(x_robot)
                loss = torch.nn.functional.l1_loss(z_student, z_teacher)
                valid_loss += loss.item()
                valid_batches += 1

        avg_train = train_loss / max(1, train_batches)
        avg_valid = valid_loss / max(1, valid_batches)

        if avg_valid < best_val:
            best_val = avg_valid
            torch.save(student.state_dict(), save_path)

        print(f"[Distill][{rkey}] Epoch {epoch}: train {avg_train:.6f}\tvalid {avg_valid:.6f}\t(best {best_val:.6f})", flush=True)

    print(f"[Distill] Saved best student encoder for {rkey} -> {save_path}", flush=True)


def code_align_phase(
    student: torch.nn.Module,
    teacher_encoder: torch.nn.Module,
    pre_quant: torch.nn.Module,
    vector_quantization: torch.nn.Module,
    train_dataloader,
    valid_dataloader,
    device: torch.device,
    save_path: str,
    epochs: int,
    lr: float,
    wd: float,
    l1_w: float,
    embed_w: float,
    rkey: str,
):
    """Code Alignment: align quantized codes between teacher and student paths."""
    ca_opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
    best_val = float('inf')
    print(f"==== [CodeAlign] Align quantized codes for {rkey} ====", flush=True)

    for epoch in range(epochs):
        # Train
        student.train()
        train_loss = 0.0
        train_batches = 0
        for x_mano, ydict in train_dataloader:
            if rkey not in ydict:
                continue
            x_mano = x_mano.to(device)
            x_robot = ydict[rkey].to(device)

            ca_opt.zero_grad(set_to_none=True)

            # Teacher path (no grad)
            with torch.no_grad():
                z_t = teacher_encoder(x_mano)                  # (B, D)
                z_t = pre_quant(z_t)                           # (B, D)
                _, z_q_t, _, _, _ = vector_quantization(z_t)   # z_q_t: (B, D)

            # Student path
            z_s = student(x_robot)                             # (B, D)
            z_s = pre_quant(z_s)                               # (B, D)
            embed_loss_s, z_q_s, _, _, _ = vector_quantization(z_s)

            l1 = torch.nn.functional.l1_loss(z_q_s, z_q_t)
            loss = l1_w * l1 + embed_w * embed_loss_s
            loss.backward()
            ca_opt.step()

            train_loss += loss.item()
            train_batches += 1

        # Valid
        student.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for x_mano, ydict in valid_dataloader:
                if rkey not in ydict:
                    continue
                x_mano = x_mano.to(device)
                x_robot = ydict[rkey].to(device)

                z_t = teacher_encoder(x_mano)
                z_t = pre_quant(z_t)
                _, z_q_t, _, _, _ = vector_quantization(z_t)

                z_s = student(x_robot)
                z_s = pre_quant(z_s)
                embed_loss_s, z_q_s, _, _, _ = vector_quantization(z_s)

                l1 = torch.nn.functional.l1_loss(z_q_s, z_q_t)
                loss = l1
                valid_loss += loss.item()
                valid_batches += 1

        avg_train = train_loss / max(1, train_batches)
        avg_valid = valid_loss / max(1, valid_batches)

        if avg_valid < best_val:
            best_val = avg_valid
            torch.save(student.state_dict(), save_path)

        print(f"[CodeAlign][{rkey}] Epoch {epoch}: train {avg_train:.6f}\tvalid {avg_valid:.6f}\t(best {best_val:.6f})", flush=True)

    print(f"[CodeAlign] Saved best student encoder for {rkey} -> {save_path}", flush=True)


def phase2_finetune(
    student: torch.nn.Module,
    state: dict,
    mano_input_length: int,
    in_dim: int,
    h_dim: int,
    n_res_layers: int,
    res_h_dim: int,
    n_embeddings: int,
    embedding_dim: int,
    beta: float,
    num_decoders: int,
    decoder_out_channels: Optional[List[int]],
    use_mlp: bool,
    train_dataloader,
    valid_dataloader,
    device: torch.device,
    save_path: str,
    epochs: int,
    lr: float,
    wd: float,
    rkey: str,
):
    """Phase 2: Freeze everything except encoder and finetune inside VQ-VAE."""
    print(f"==== [Phase 2] Fine-tune student encoder inside VQ-VAE for {rkey} ====", flush=True)
    vqvae_ft = MultiDecoderVQVAE(
        in_dim=in_dim,
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        n_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        beta=beta,
        num_decoders=num_decoders,
        decoder_out_channels=decoder_out_channels,
        use_mlp=use_mlp,
        input_length=mano_input_length,
    ).to(device)
    vqvae_ft.load_state_dict(state)

    # Replace encoder with the (distilled/aligned) student
    vqvae_ft.encoder = student

    # Freeze everything except encoder
    for _, p in vqvae_ft.named_parameters():
        p.requires_grad = False
    for p in vqvae_ft.encoder.parameters():
        p.requires_grad = True

    ft_opt = torch.optim.AdamW(vqvae_ft.encoder.parameters(), lr=lr, weight_decay=wd)
    ft_best = float('inf')
    n_e = vqvae_ft.vector_quantization.n_e

    for epoch in range(epochs):
        # Train
        vqvae_ft.train()
        train_loss = 0.0
        train_loss_l1 = 0.0
        train_batches = 0
        train_used = torch.zeros(n_e, dtype=torch.bool)
        for _, ydict in train_dataloader:
            if rkey not in ydict:
                continue
            x_robot = ydict[rkey].to(device)

            ft_opt.zero_grad(set_to_none=True)
            embedding_loss, outs, ppl = vqvae_ft(x_robot, return_all=True)
            if getattr(vqvae_ft, 'last_code_indices', None) is not None:
                train_used[vqvae_ft.last_code_indices] = True
            l1_ = compute_loss(outs, ydict)
            loss = l1_ + embedding_loss
            loss.backward()
            ft_opt.step()

            train_loss += loss.item()
            train_loss_l1 += l1_.item()
            train_batches += 1

        # Valid
        vqvae_ft.eval()
        valid_loss = 0.0
        valid_loss_l1 = 0.0
        valid_batches = 0
        valid_used = torch.zeros(n_e, dtype=torch.bool)
        with torch.no_grad():
            for _, ydict in valid_dataloader:
                if rkey not in ydict:
                    continue
                x_robot = ydict[rkey].to(device)
                embedding_loss, outs, ppl = vqvae_ft(x_robot, return_all=True)
                if getattr(vqvae_ft, 'last_code_indices', None) is not None:
                    valid_used[vqvae_ft.last_code_indices] = True
                l1_ = compute_loss(outs, ydict)
                loss = l1_ + embedding_loss
                valid_loss += loss.item()
                valid_loss_l1 += l1_.item()
                valid_batches += 1

        avg_train = train_loss / max(1, train_batches)
        avg_valid = valid_loss / max(1, valid_batches)
        avg_valid_l1 = valid_loss_l1 / max(1, valid_batches)

        train_used_cnt = int(train_used.sum().item())
        valid_used_cnt = int(valid_used.sum().item())
        train_util = train_used_cnt / n_e if n_e > 0 else 0.0
        valid_util = valid_used_cnt / n_e if n_e > 0 else 0.0

        if avg_valid_l1 < ft_best:
            ft_best = avg_valid_l1
            torch.save(vqvae_ft.encoder.state_dict(), save_path)

        print(f"[Finetune][{rkey}] Epoch {epoch}: train {avg_train:.6f}\tvalid {avg_valid:.6f}\t(best {ft_best:.6f})", flush=True)
        print(f"    l1 Loss (train): {train_loss_l1/max(1, train_batches):.6f}\t(valid): {valid_loss_l1/max(1, valid_batches):.6f}", flush=True)
        print(f"    Codebook Util (train): {train_used_cnt}/{n_e} ({train_util*100:.1f}%)\t(valid): {valid_used_cnt}/{n_e} ({valid_util*100:.1f}%)", flush=True)

    print(f"[Finetune] Overwrote best student encoder for {rkey} -> {save_path}", flush=True)
    del vqvae_ft


def build_argparser():
    p = argparse.ArgumentParser(description="Train multi-encoder student with optional phases")

    # IO/config
    p.add_argument("--dataset", type=str, default="/home/main/dex-ICLR/UniHM/UniHM/dataset/dataset.npz", help="Path to dataset .npz")
    p.add_argument("--config", type=str, default="/home/main/dex-ICLR/UniHM/UniHM/ckpt/memd/conv1d/config.json", help="Path to VQ-VAE config json")
    p.add_argument("--ckpt", type=str, default="/home/main/dex-ICLR/UniHM/UniHM/ckpt/memd/conv1d/memd_conv1d.pth", help="Path to VQ-VAE checkpoint")
    p.add_argument("--save_dir", type=str, default="/home/main/dex-ICLR/UniHM/UniHM/ckpt/memd/conv1d/encoders", help="Directory to save student encoders")

    # Data
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--val_batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Device
    p.add_argument("--device", type=str, default=("cuda:1" if torch.cuda.is_available() else "cpu"))

    # Phases toggles
    p.add_argument("--do_phase1", action="store_true",default=True, help="Run Phase 1: distillation to teacher encoder")
    p.add_argument("--do_code_align", action="store_true",default=True, help="Run Code Alignment phase between Phase 1 and 2")
    p.add_argument("--do_phase2", action="store_true",default=True, help="Run Phase 2: encoder-only finetune inside VQ-VAE")

    # Phase 1 (distill) hparams
    p.add_argument("--phase1_epochs", type=int, default=100)
    p.add_argument("--phase1_lr", type=float, default=5e-3)
    p.add_argument("--phase1_wd", type=float, default=1e-4)

    # Code alignment hparams
    p.add_argument("--code_align_epochs", type=int, default=100)
    p.add_argument("--code_align_lr", type=float, default=1e-3)
    p.add_argument("--code_align_wd", type=float, default=1e-4)
    p.add_argument("--code_align_l1_w", type=float, default=1.0, help="Weight for L1 loss between quantized codes")
    p.add_argument("--code_align_embed_w", type=float, default=1.0, help="Weight for embedding loss from VectorQuantizer on student path")

    # Phase 2 (finetune) hparams
    p.add_argument("--phase2_epochs", type=int, default=20)
    p.add_argument("--phase2_lr", type=float, default=5e-4)
    p.add_argument("--phase2_wd", type=float, default=1e-4)

    # Optional: run only specific robots
    p.add_argument("--robot_keys", type=str, default="", help="Comma-separated robot keys to train (override autodetect)")

    return p


def make_student_encoder(use_mlp: bool, r_in_len: int, h_dim: int, n_res_layers: int, res_h_dim: int, embedding_dim: int, in_dim: int):
    if use_mlp:
        return MLPEncoder(r_in_len, h_dim, n_res_layers, res_h_dim, embedding_dim=embedding_dim)
    else:
        return Encoder(in_dim, h_dim, n_res_layers, res_h_dim, x_shape=r_in_len, embedding_dim=embedding_dim)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace):
    global device
    device = torch.device(args.device)

    set_seed(args.seed)

    # 1) Load dataset
    print("load dataset from", args.dataset)
    data = load_dataset_single(args.dataset)

    random.shuffle(data)
    train_dataset = HandDataset(data)
    val_dataset = HandDataset(data)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        persistent_workers=(args.num_workers > 0),
    )
    valid_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 4),
        pin_memory=("cuda" in args.device),
        persistent_workers=(args.num_workers > 0),
    )

    # 2) Load VQ-VAE from config and checkpoint to get teacher encoder
    print("load vqvae from", args.config, args.ckpt)
    config = json.load(open(args.config, "r"))
    in_dim = config.get("in_dim")
    h_dim = config.get("h_dim")
    res_h_dim = config.get("res_h_dim")
    n_res_layers = config.get("n_res_layers")
    n_embeddings = config.get("n_embeddings")
    embedding_dim = config.get("embedding_dim")
    beta = config.get("beta")
    num_decoders = config.get("num_decoders")
    decoder_out_channels: Optional[List[int]] = config.get("decoder_out_channels")
    use_mlp = config.get("use_mlp")
    input_length = config.get("input_length")

    # Determine input vector length (mano hand pose length)
    x0, _ = train_dataset[0]
    mano_input_length = int(x0.shape[-1])

    model = MultiDecoderVQVAE(
        in_dim=in_dim,
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        n_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        beta=beta,
        num_decoders=num_decoders,
        decoder_out_channels=decoder_out_channels,
        use_mlp=use_mlp,
        input_length=mano_input_length,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    # Teacher encoder: freeze
    teacher_encoder: torch.nn.Module = model.encoder
    teacher_encoder.eval()
    for p in teacher_encoder.parameters():
        p.requires_grad = False

    # Components for code alignment (shared, frozen)
    pre_quant = model.pre_quantization_conv
    vector_quantization = model.vector_quantization
    for p in pre_quant.parameters():
        p.requires_grad = False
    for p in vector_quantization.parameters():
        p.requires_grad = False
    pre_quant.eval()
    vector_quantization.eval()

    # 3) Discover available robot keys in dataset sample and their input lengths
    sample = data[0]
    # Prefer ordered keys in ROBOT_KEYS_ORDER, then append any remaining keys ending with _qpos
    present_robot_keys = [k for k in ROBOT_KEYS_ORDER if k in sample]
    remaining = [k for k in sample.keys() if k.endswith('_qpos') and k not in present_robot_keys]
    robot_keys = present_robot_keys + remaining

    # Override by args if provided
    if args.robot_keys:
        wanted = [k.strip() for k in args.robot_keys.split(',') if k.strip()]
        robot_keys = [k for k in robot_keys if k in wanted]

    os.makedirs(args.save_dir, exist_ok=True)

    # 4) Train a student encoder for each robot sequentially
    print("train robots:", robot_keys)
    for rkey in robot_keys:
        # Input length for this robot
        r_in_len = int(sample[rkey].shape[0])
        # Create student encoder with same bottleneck size
        student = make_student_encoder(use_mlp, r_in_len, h_dim, n_res_layers, res_h_dim, embedding_dim, in_dim).to(device)

        save_path = os.path.join(args.save_dir, f"student_encoder_{rkey}.pth")

        # =============================
        # Phase 1: Distillation to teacher encoder (optional)
        # =============================
        if args.do_phase1:
            phase1_distill(
                student=student,
                teacher_encoder=teacher_encoder,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                device=device,
                save_path=save_path,
                epochs=args.phase1_epochs,
                lr=args.phase1_lr,
                wd=args.phase1_wd,
                rkey=rkey,
                r_in_len=r_in_len,
            )
        else:
            # Try to warm start from disk if phase1 skipped
            if os.path.exists(save_path):
                student.load_state_dict(torch.load(save_path, map_location=device))
                print(f"[Phase 1 skipped] Loaded existing weights for {rkey} from {save_path}", flush=True)
            else:
                print(f"[Phase 1 skipped] No existing weights found for {rkey}, training will proceed from scratch in later phases.", flush=True)

        # Load best (or existing) weights before next phases
        try:
            if os.path.exists(save_path):
                student.load_state_dict(torch.load(save_path, map_location=device))
        except Exception as e:
            print(f"Warning: failed to reload weights for {rkey}: {e}", flush=True)

        # =============================
        # Code Alignment phase (optional)
        # =============================
        if args.do_code_align:
            code_align_phase(
                student=student,
                teacher_encoder=teacher_encoder,
                pre_quant=pre_quant,
                vector_quantization=vector_quantization,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                device=device,
                save_path=save_path,
                epochs=args.code_align_epochs,
                lr=args.code_align_lr,
                wd=args.code_align_wd,
                l1_w=args.code_align_l1_w,
                embed_w=args.code_align_embed_w,
                rkey=rkey,
            )
        else:
            print(f"[CodeAlign skipped] for {rkey}", flush=True)

        # Load best weights before finetune
        try:
            if os.path.exists(save_path):
                student.load_state_dict(torch.load(save_path, map_location=device))
        except Exception as e:
            print(f"Warning: failed to reload code-align weights for {rkey}: {e}", flush=True)

        ### ============================
        for i in range(4):
            try:
                if os.path.exists(save_path):
                    student.load_state_dict(torch.load(save_path, map_location=device))
            except Exception as e:
                print(f"Warning: failed to reload code-align weights for {rkey}: {e}", flush=True)
            if args.do_phase1:
                phase1_distill(
                    student=student,
                    teacher_encoder=teacher_encoder,
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    save_path=save_path,
                    epochs=args.phase1_epochs,
                    lr=args.phase1_lr*(0.8**i),
                    wd=args.phase1_wd,
                    rkey=rkey,
                    r_in_len=r_in_len,
                )
            try:
                if os.path.exists(save_path):
                    student.load_state_dict(torch.load(save_path, map_location=device))
            except Exception as e:
                print(f"Warning: failed to reload code-align weights for {rkey}: {e}", flush=True)
            if args.do_code_align:
                code_align_phase(
                    student=student,
                    teacher_encoder=teacher_encoder,
                    pre_quant=pre_quant,
                    vector_quantization=vector_quantization,
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    save_path=save_path,
                    epochs=args.code_align_epochs,
                    lr=args.code_align_lr*(0.8**i),
                    wd=args.code_align_wd,
                    l1_w=args.code_align_l1_w,
                    embed_w=args.code_align_embed_w,
                    rkey=rkey,
                )
        ########

        # =============================
        # Phase 2: Fine-tune within VQ-VAE (optional)
        # =============================
        if args.do_phase2:
            phase2_finetune(
                student=student,
                state=state,
                mano_input_length=mano_input_length,
                in_dim=in_dim,
                h_dim=h_dim,
                n_res_layers=n_res_layers,
                res_h_dim=res_h_dim,
                n_embeddings=n_embeddings,
                embedding_dim=embedding_dim,
                beta=beta,
                num_decoders=num_decoders,
                decoder_out_channels=decoder_out_channels,
                use_mlp=use_mlp,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                device=device,
                save_path=save_path,
                epochs=args.phase2_epochs,
                lr=args.phase2_lr,
                wd=args.phase2_wd,
                rkey=rkey,
            )
        else:
            print(f"[Phase 2 skipped] for {rkey}", flush=True)

        del student
        torch.cuda.empty_cache()


def main():
    args = build_argparser().parse_args()

    # If user didn't pass any toggles, default to running all phases to preserve old behavior
    if not (args.do_phase1 or args.do_code_align or args.do_phase2):
        args.do_phase1 = True
        args.do_code_align = True
        args.do_phase2 = True

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()