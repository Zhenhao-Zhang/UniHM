from UniHM.dataset import HandDataset,load_dataset_single
from UniHM.vqvae import MultiDecoderVQVAE
import torch
import random
import matplotlib.pyplot as plt

ROBOT_KEYS_ORDER = [
    "allegro_hand_qpos",
    "shadow_hand_qpos",
    "svh_hand_qpos",
    "leap_hand_qpos",
    "ability_hand_qpos",
    "panda_hand_qpos",
]


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# KMeans reset configuration
KMEANS_INTERVAL = 40  # every N epochs
KMEANS_ITERS = 10
KMEANS_MAX_SAMPLES = 50000
KMEANS_MAX_BATCHES = 40
KMEANS_FIRST_EPOCH = 5          # trigger an early KMeans reset at this epoch (1-based)
KMEANS_MAX_RESET_EPOCH = 10000   # no KMeans reset will occur after this epoch (1-based)
# KMEANS_ONLY_UPDATE_COLD_EPOCH = 120  # after this epoch, only update cold codes during KMeans reset

def compute_loss(ypred, ydict):
    def loss_func(x, y):
        return torch.nn.functional.l1_loss(x, y)
    ylist = []
    for k in ROBOT_KEYS_ORDER:
        if k in ydict:
            ylist.append(ydict[k])
    losses = [loss_func(x, y.to(device)) for x,y in zip(ypred,ylist)]
    return sum(losses)


def train():
    data=load_dataset_single('/home/main/dex-ICLR/UniHM/UniHM/dataset/dataset.npz')
    ckpt= "/home/main/dex-ICLR/UniHM/model_without_cold.pth"
    random.shuffle(data)
    train_dataset = HandDataset(data)
    val_dataset = HandDataset(data)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=1,
    )

    # Determine decoder output dimensions per robot
    sample = data[0]
    present_robot_keys = [k for k in ROBOT_KEYS_ORDER if k in sample]
    out_dims = [int(sample[k].shape[0]) for k in present_robot_keys]

    # Determine input vector length for encoder (L)
    x0, _ = train_dataset[0]
    input_length = int(x0.shape[-1])

    # Use MLP frontend for efficiency: (B, 1, L) -> (B, D) -> code (B, D) -> decoders (B, out_dim)
    model = MultiDecoderVQVAE(
        in_dim=1,               # kept for Conv1d compatibility; unused in MLP path
        h_dim=128,              # hidden width
        res_h_dim=128,          # residual hidden width
        n_res_layers=2,         # residual layers
        n_embeddings=8192,      # codebook size
        embedding_dim=512,     # bottleneck/code dimension D
        beta=0.25,              # VQ commitment cost
        num_decoders=len(out_dims),
        decoder_out_channels=out_dims,
        use_mlp=True,           # enable MLP path for speed and correct shapes
        input_length=input_length,
    ).to(device)
    params = {
        "in_dim": 1,
        "h_dim": 128,
        "res_h_dim": 128,
        "n_res_layers": 2,
        "n_embeddings": 8192,
        "embedding_dim": 512,
        "beta": 0.25,
        "num_decoders": len(out_dims),
        "decoder_out_channels": out_dims,
        "use_mlp": True,
        "input_length": input_length,
    }
    print("Model parameters:", params)
    try:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model from {ckpt}")
    except Exception as e:
        print(f"Failed to load model from {ckpt}: {e}")

    # Optimizers: AdamW for encoder/decoder; SGD for codebook
    encdec_params = (
        list(model.encoder.parameters())
        + list(model.pre_quantization_conv.parameters())
        + list(model.decoders.parameters())
    )
    codebook_params = list(model.vector_quantization.parameters())

    opt_encdec = torch.optim.AdamW(encdec_params, lr=5e-4, weight_decay=1e-4)
    opt_code = torch.optim.SGD(codebook_params, lr=1e-3, momentum=0.9)

    loss_best = 10000
    for epoch in range(10000):
        model.train()
        train_loss = 0
        valid_loss = 0

        # Track codebook utilization per epoch
        n_e = model.vector_quantization.n_e
        train_used = torch.zeros(n_e, dtype=torch.bool)
        valid_used = torch.zeros(n_e, dtype=torch.bool)

        # ---- Train ----
        for x, ydict in train_dataloader:
            # x is expected (B, L) or (B, 1, L); model.encode handles both
            x = x.to(device)
            opt_encdec.zero_grad(set_to_none=True)
            opt_code.zero_grad(set_to_none=True)

            embedding_loss, outs, ppl = model(x, return_all=True)
            if model.last_code_indices is not None:
                train_used[model.last_code_indices] = True

            loss = compute_loss(outs, ydict) + embedding_loss
            loss.backward()
            opt_encdec.step()
            opt_code.step()
            train_loss += loss.item()

        # ---- Valid ----
        model.eval()
        with torch.no_grad():
            for x, ydict in valid_dataloader:
                x = x.to(device)
                embedding_loss, outs, ppl = model(x, return_all=True)
                if model.last_code_indices is not None:
                    valid_used[model.last_code_indices] = True
                loss = compute_loss(outs, ydict) + embedding_loss
                valid_loss += loss.item()

        # compute utilization stats
        train_used_cnt = int(train_used.sum().item())
        valid_used_cnt = int(valid_used.sum().item())
        train_util = train_used_cnt / n_e
        valid_util = valid_used_cnt / n_e

        if valid_loss < loss_best:
            loss_best = valid_loss
            torch.save(model.state_dict(), "model_without_cold.pth")
        print(f"Epoch {epoch}\tLoss: {train_loss/len(train_dataloader):.6f}\tValid Loss: {valid_loss/len(valid_dataloader):.6f}")
        print(f"\tCodebook Util (train): {train_used_cnt}/{n_e} ({train_util*100:.1f}%)\t(valid): {valid_used_cnt}/{n_e} ({valid_util*100:.1f}%)")

        # ---- Periodic KMeans reset of codebook ----
        current_epoch = epoch + 1  # convert to 1-based for readability
        should_reset_first = current_epoch == KMEANS_FIRST_EPOCH
        if current_epoch < 300:
            flag_only_update_cold = False  # set True to update only cold codes after warmup
        else:
            flag_only_update_cold = True
        if current_epoch < 500:
            should_reset_regular = (current_epoch % KMEANS_INTERVAL == 0)
        else:
            should_reset_regular = (current_epoch % (KMEANS_INTERVAL*2) == 0)

        within_reset_window = current_epoch <= KMEANS_MAX_RESET_EPOCH
        if (should_reset_first or should_reset_regular) and within_reset_window:
            zs = []
            with torch.no_grad():
                model.eval()
                for i, (x, _) in enumerate(train_dataloader):
                    if i >= KMEANS_MAX_BATCHES:
                        break
                    x = x.to(device)
                    # Encode to (B, D)
                    z_e = model.encode(x)  # (B, D)
                    zs.append(z_e.detach().cpu())
            if zs:
                ze_data = torch.cat(zs, dim=0)
                print(f"[KMeans] resetting codebook on {ze_data.shape[0]} samples ... only update cold codes: {flag_only_update_cold}")
                model.vector_quantization.kmeans_reset_codebook(
                    ze_data,
                    num_iters=KMEANS_ITERS,
                    max_samples=KMEANS_MAX_SAMPLES,
                    seed=42,
                    verbose=False,
                    update_only_cold=flag_only_update_cold,
                )
                print("[KMeans] codebook reset done.")

if __name__ == "__main__":
    train()