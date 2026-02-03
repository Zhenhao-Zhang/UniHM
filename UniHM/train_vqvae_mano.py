import os
import json
import argparse
import torch
import torch.nn as nn
from typing import List
from UniHM.dataset import HandDataset, load_dataset_single
from UniHM.vqvae import MultiDecoderVQVAE
from UniHM.vqvae.decoder import Decoder, MLPDecoder
import random

ROBOT_KEYS_ORDER = [
    "allegro_hand_qpos",
    "shadow_hand_qpos",
    "svh_hand_qpos",              # may not exist (dataset uses schunk_svh_hand_qpos)
    "schunk_svh_hand_qpos",       # add alternative key to be safe
    "leap_hand_qpos",
    "ability_hand_qpos",
    "panda_hand_qpos",
    "panda_gripper_qpos",         # fallback if naming differs
]

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_with_mano_head(old_params:dict, mano_dim:int, robot_out_dims:List[int], ckpt_path:str, device:torch.device):
    """Build base model (robot decoders only), then attach a separate MANO decoder as model.mano_decoder."""
    model = MultiDecoderVQVAE(
        in_dim=old_params.get("in_dim", 1),
        h_dim=old_params.get("h_dim", 128),
        res_h_dim=old_params.get("res_h_dim", 128),
        n_res_layers=old_params.get("n_res_layers", 2),
        n_embeddings=old_params.get("n_embeddings", 8192),
        embedding_dim=old_params.get("embedding_dim", 512),
        beta=old_params.get("beta", 0.25),
        num_decoders=len(robot_out_dims),
        decoder_out_channels=robot_out_dims,
        use_mlp=old_params.get("use_mlp", False),
        input_length=old_params.get("input_length"),
    ).to(device)
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded base ckpt: {ckpt_path}\n  Missing: {missing}\n  Unexpected: {unexpected}")
    else:
        print(f"Warning: ckpt {ckpt_path} not found; base model random.")

    # Create MANO decoder head (separate attribute)
    if old_params.get("use_mlp", False):
        mano_dec = MLPDecoder(old_params.get("embedding_dim", 512),
                              old_params.get("h_dim", 128),
                              old_params.get("n_res_layers", 2),
                              old_params.get("res_h_dim", 128),
                              out_channels=mano_dim)
    else:
        mano_dec = Decoder(old_params.get("in_dim", 1),
                           old_params.get("h_dim", 128),
                           old_params.get("n_res_layers", 2),
                           old_params.get("res_h_dim", 128),
                           outdim=mano_dim,
                           embedding_dim=old_params.get("embedding_dim", 512))
    model.mano_decoder = mano_dec.to(device)
    return model


def train_mano_decoder(args:argparse.Namespace):
    device = torch.device(args.device)
    set_seed(args.seed)

    # Load dataset
    print("Loading dataset:", args.dataset)
    data = load_dataset_single(args.dataset)
    random.shuffle(data)
    train_dataset = HandDataset(data)
    val_dataset = HandDataset(data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=('cuda' in args.device)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=max(1, args.num_workers//4), pin_memory=('cuda' in args.device)
    )

    # Determine MANO pose dimension
    x0, _ = train_dataset[0]
    mano_dim = int(x0.shape[-1])

    # Determine existing robot decoder output dims
    sample = data[0]
    present_robot_keys = [k for k in ROBOT_KEYS_ORDER if k in sample and k.endswith('qpos')]
    robot_out_dims = [int(sample[k].shape[0]) for k in present_robot_keys]
    print("Robot keys (frozen):", present_robot_keys)
    print("Robot decoder dims:", robot_out_dims, "MANO dim:", mano_dim)

    # Hyperparams / config
    params = {
        "in_dim": args.in_dim,
        "h_dim": args.h_dim,
        "res_h_dim": args.res_h_dim,
        "n_res_layers": args.n_res_layers,
        "n_embeddings": args.n_embeddings,
        "embedding_dim": args.embedding_dim,
        "beta": args.beta,
        "use_mlp": args.use_mlp,
        "input_length": mano_dim,
    }
    if args.config and os.path.isfile(args.config):
        try:
            cfg = json.load(open(args.config, 'r'))
            params.update({k: cfg.get(k, v) for k,v in params.items()})
            print("Loaded params from config:", args.config)
        except Exception as e:
            print("Failed to read config, using CLI params:", e)

    model = build_model_with_mano_head(params, mano_dim, robot_out_dims, args.ckpt, device)

    # Freeze everything except mano_decoder
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mano_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.mano_decoder.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val = float('inf')
    l1 = nn.L1Loss()

    for epoch in range(args.epochs):
        # Train
        model.train(); model.mano_decoder.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z_e = model.encode(x)
                _, z_q, _, _, _ = model.quantize(z_e)
            y_hat = model.mano_decoder(z_q).squeeze(-1)
            loss = l1(y_hat, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()

        # Valid
        model.eval(); model.mano_decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                z_e = model.encode(x)
                _, z_q, _, _, _ = model.quantize(z_e)
                y_hat = model.mano_decoder(z_q).squeeze(-1)
                val_loss += l1(y_hat, x).item()
        tr_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
            torch.save(model.state_dict(), args.out_ckpt)
            torch.save(model.mano_decoder.state_dict(), args.out_decoder)
            print(f"[Epoch {epoch}] BEST val {val_loss:.6f} (train {tr_loss:.6f}) -> saved")
        else:
            print(f"[Epoch {epoch}] train {tr_loss:.6f} val {val_loss:.6f} (best {best_val:.6f})")

    print("Done. Best val:", best_val)


def parse_args():
    p = argparse.ArgumentParser("Train separate MANO decoder on existing VQ-VAE latent space")
    p.add_argument('--dataset', type=str, default='/home/main/dex-ICLR/UniHM/UniHM/dataset/dataset.npz')
    p.add_argument('--ckpt', type=str, default='/home/main/dex-ICLR/UniHM/UniHM/ckpt/memd/conv1d/memd_conv1d.pth')
    p.add_argument('--config', type=str, default='')
    p.add_argument('--out_ckpt', type=str, default='/home/main/dex-ICLR/UniHM/UniHM/ckpt/memd/conv1d/memd_conv1d_mano_decoder.pth')
    p.add_argument('--out_decoder', type=str, default='/home/main/dex-ICLR/UniHM/UniHM/ckpt/memd/conv1d/mano_decoder_only.pth')
    p.add_argument('--in_dim', type=int, default=1)
    p.add_argument('--h_dim', type=int, default=128)
    p.add_argument('--res_h_dim', type=int, default=128)
    p.add_argument('--n_res_layers', type=int, default=2)
    p.add_argument('--n_embeddings', type=int, default=8192)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--use_mlp', action='store_true', default=False)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--val_batch_size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=('cuda:0' if torch.cuda.is_available() else 'cpu'))
    return p.parse_args()


def main():
    args = parse_args(); train_mano_decoder(args)

if __name__ == '__main__':
    main()
