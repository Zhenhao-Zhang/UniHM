import numpy as np
from glob import glob
import os
from scipy import linalg
import numpy as np
from scipy.spatial import cKDTree, Delaunay
import tqdm

def load_data(file):
    """Load per-sample npz produced by forward.py including new generation2 and generation_sim.

    Returns:
        gt_data: dict with GT joint sequences
        network_data: first generation
        network2_data: second generation (same text)
        network_sim_data: similar text generation
        optimization_data: optimized trajectories
    """
    data = dict(np.load(file, allow_pickle=True)["data"].tolist())
    # grasped_obj_xyz = data["object_mesh_file"][data["grasped_obj_idx"]]
    try:
        grasped_obj_xyz = [os.path.join(os.path.dirname(obj), "points.xyz") for obj in data["object_mesh_file"]][data["grasped_obj_idx"]]
        grasped_obj_point3d = np.loadtxt(grasped_obj_xyz)
    except:
        grasped_obj_xyz = data["object_mesh_file"][data["grasped_obj_idx"]]
        grasped_obj_point3d = np.loadtxt(grasped_obj_xyz)
    # Fallback handling for possible key naming differences
    raw_block = data.get("raw", {})
    gen_block = data.get("generation", {})
    gen2_block = data.get("generation2", {})
    gensim_block = data.get("generation_sim", {})
    opt_block = data.get("optimization", {})

    def _maybe(keys, block):
        for k in keys:
            if k in block:
                return np.array(block[k])
        # Return empty array if missing
        return np.zeros((0,))

    gt_data = {
        "objpose": np.array(data["grasped_obj_pose"]),  # (T,7)
        "objpointcloud": grasped_obj_point3d,  # (N,3)
        "allegro": _maybe(["allegro"], raw_block),
        "shadow": _maybe(["shadow"], raw_block),
        "svh": _maybe(["svh", "schunk_svh"], raw_block),
    }
    network_data = {
        "allegro": _maybe(["allegro"], gen_block),
        "shadow": _maybe(["shadow"], gen_block),
        "svh": _maybe(["svh", "schunk_svh"], gen_block),
    }
    network2_data = {
        "allegro": _maybe(["allegro"], gen2_block),
        "shadow": _maybe(["shadow"], gen2_block),
        "svh": _maybe(["svh", "schunk_svh"], gen2_block),
    }
    network_sim_data = {
        "allegro": _maybe(["allegro"], gensim_block),
        "shadow": _maybe(["shadow"], gensim_block),
        "svh": _maybe(["svh", "schunk_svh"], gensim_block),
    }
    optimization_data = {
        "allegro": _maybe(["allegro"], opt_block),
        "shadow": _maybe(["shadow"], opt_block),
        "svh": _maybe(["svh", "schunk_svh"], opt_block),
    }
    return gt_data, network_data, network2_data, network_sim_data, optimization_data


def mpjpe(pred, gt):
    """Mean Per Joint Position Error.
    Same as before: ignore first 6 dims (e.g., global pose) and truncate to 72 frames.
    """
    if pred.ndim == 3:
        pred = pred.mean(0)
    if gt.ndim == 3:
        gt = gt.mean(0)
    T = min(72, getattr(pred, 'shape', [0])[0], getattr(gt, 'shape', [0])[0])
    if T == 0:
        return 0.0
    diff = np.abs(pred[:T, 6:] - gt[:T, 6:])
    return float(diff.sum() / T)

def fhlt(pred, gt):
    """First-Hand Left-Translation error (FH-LT): mean absolute diff over first 3 dims (<=72 frames)."""
    if pred.ndim == 3:
        pred = pred.mean(0)
    if gt.ndim == 3:
        gt = gt.mean(0)
    T = min(72, getattr(pred, 'shape', [0])[0], getattr(gt, 'shape', [0])[0])
    if T == 0:
        return 0.0
    diff = np.abs(pred[:T, :3] - gt[:T, :3])
    return float(diff.sum() / T)

def fhlr(pred, gt):
    """First-Hand Left-Rotation error (FH-LR): mean absolute diff over dims 3:6 (<=72 frames)."""
    if pred.ndim == 3:
        pred = pred.mean(0)
    if gt.ndim == 3:
        gt = gt.mean(0)
    T = min(72, getattr(pred, 'shape', [0])[0], getattr(gt, 'shape', [0])[0])
    if T == 0:
        return 0.0
    diff = np.abs(pred[:T, 3:6] - gt[:T, 3:6])
    return float(diff.sum() / T)

def fid(pred, gt, eps=1e-6):
    if pred.ndim == 3:
        # 合并 (N,T,D) -> (N*T, D)
        pred_flat = pred.reshape(-1, pred.shape[-1])
    else:
        pred_flat = pred
    if gt.ndim == 3:
        gt_flat = gt.reshape(-1, gt.shape[-1])
    else:
        gt_flat = gt
    mu1 = np.mean(pred_flat, axis=0)
    mu2 = np.mean(gt_flat, axis=0)
    sig1 = np.cov(pred_flat, rowvar=False)
    sig2 = np.cov(gt_flat, rowvar=False)
    # 数值稳定
    if sig1.shape == sig2.shape:
        sig1 += np.eye(sig1.shape[0]) * eps
        sig2 += np.eye(sig2.shape[0]) * eps
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = np.eye(sig1.shape[0])
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_value = np.sum((mu1 - mu2)**2) + np.trace(sig1 + sig2 - 2*covmean)
    return float(fid_value)

def _aligned_truncate(a, b):
    """Truncate two (T,D) arrays to same min length (<=72). Return views."""
    T = min(72, a.shape[0], b.shape[0])
    if T == 0:
        return a[:0], b[:0]
    return a[:T], b[:T]

def diversity(gen_sim, gen):
    """Diversity metric: L2 distance per frame between similar-text generation and original generation (<=72 frames)."""
    if gen.ndim == 3:
        gen = gen.mean(0)
    if gen_sim.ndim == 3:
        gen_sim = gen_sim.mean(0)
    gs, g = _aligned_truncate(gen_sim, gen)
    if g.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(gs - g, axis=-1).mean())*16

METRIC_FUNCS = {
    "mpjpe": mpjpe,
    "fhlt": fhlt,
    "fhlr": fhlr,
    "fid": fid,
}

def compute_gt_diversity_between_samples(seq_list):
    """GT diversity across samples (NOT intra-sample).

    Given a list of sequences (each (T,D) or (N,T,D)), compute the mean absolute
    difference between every pair of consecutive samples' sequences:
        avg_{i} mean_{t,d} | sample_{i+1}(t,d) - sample_i(t,d) |
    Each pair is aligned by:
      * If ndim==3, average over first dim -> (T,D)
      * Truncate both to same min length <=72 frames
      * If truncated length < 1, skip pair
    Returns 0.0 if <2 usable samples.
    """
    if len(seq_list) < 2:
        return 0.0
    pair_vals = []
    for a, b in zip(seq_list[:-1], seq_list[1:]):
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            continue
        if a.ndim == 3:
            a = a.mean(0)
        if b.ndim == 3:
            b = b.mean(0)
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            continue
        T = min(72, a.shape[0], b.shape[0])
        if T == 0:
            continue
        aa = a[:T]
        bb = b[:T]
        diff = np.abs(bb - aa).mean()
        pair_vals.append(diff)
    if not pair_vals:
        return 0.0
    return float(np.mean(pair_vals))*16

def _ensure_samples(x):
    """如果是 (T,D) => (1,T,D) 方便统一处理。"""
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return x[None, ...]
    return x

def compute_pair_metrics(pred, gt):
    """Compute metrics comparing prediction to ground truth."""
    results = {}
    for name, fn in METRIC_FUNCS.items():
        try:
            if name == "fid":
                results[name] = fn(pred, gt)
            else:
                results[name] = fn(pred, gt)
        except Exception:
            results[name] = float('nan')
    return results

HANDS = ["allegro", "shadow", "svh"]

def evaluate_file(file_path):
    gt_data, net_data, net2_data, net_sim_data, opt_data = load_data(file_path)
    per_hand = {}
    gt_sequences = {}
    for hand in HANDS:
        gt = net = net_sim = opt = None
        try:
            gt = gt_data[hand]
            net = net_data[hand]
            net_sim = net_sim_data[hand]
            opt = opt_data[hand]
        except KeyError:
            continue
        gt_sequences[hand] = gt
        net_pair = compute_pair_metrics(net, gt)
        opt_pair = compute_pair_metrics(opt, gt)
        # Updated diversity (similar-text vs original)
        try:
            net_pair["diversity"] = diversity(net_sim, net)
        except Exception:
            net_pair["diversity"] = float('nan')
        per_hand[hand] = {
            "network": net_pair,
            "optimization": opt_pair,
        }
    return per_hand, gt_sequences

def aggregate(all_file_metrics):
    # all_file_metrics: list[per_hand dict]
    # 结构: metrics[hand][mode][metric]
    agg = {}
    for hand in HANDS:
        agg[hand] = {"network": {}, "optimization": {}}
    # 收集所有 metric 名称
    metric_names = set()
    for fm in all_file_metrics:
        for hand in HANDS:
            for mode in ["network", "optimization"]:
                metric_names.update(fm[hand][mode].keys())
    for hand in HANDS:
        for mode in ["network", "optimization"]:
            for m in metric_names:
                values = []
                for fm in all_file_metrics:
                    if m in fm[hand][mode] and not np.isnan(fm[hand][mode][m]):
                        values.append(fm[hand][mode][m])
                if values:
                    agg[hand][mode][m] = float(np.mean(values))
                else:
                    agg[hand][mode][m] = float('nan')
    # 计算所有手的平均(宏平均)
    macro = {"network": {}, "optimization": {}}
    for mode in ["network", "optimization"]:
        for m in metric_names:
            vals = [agg[h][mode][m] for h in HANDS if not np.isnan(agg[h][mode][m])]
            if vals:
                macro[mode][m] = float(np.mean(vals))
            else:
                macro[mode][m] = float('nan')
    return agg, macro

def print_results(title, agg, macro):
    print(f"===== {title} 结果 =====")
    scale = 180/6.28
    for hand in HANDS:
        print(f"--- {hand} ---")
        for mode in ["network", "optimization"]:
            metrics_items = []
            for k, v in sorted(agg[hand][mode].items()):
                if k in ('fhlt'):
                    display_v = v * 100 if k == 'fhlt' else v
                else:
                    display_v = v * scale
                metrics_items.append(f"{k}: {display_v:.4f}")
            metrics_str = ", ".join(metrics_items)
            print(f"  {mode}: {metrics_str}")
    print("--- 宏平均(所有手) ---")
    for mode in ["network", "optimization"]:
        metrics_items = []
        for k, v in sorted(macro[mode].items()):
            if k in ('fhlt'):
                display_v = v * 100 if k == 'fhlt' else v
            else:
                display_v = v * scale
            metrics_items.append(f"{k}: {display_v:.4f}")
        metrics_str = ", ".join(metrics_items)
        print(f"  {mode}: {metrics_str}")
    print()

def main():
    seen_files = glob("/your_path/*.npz")
    unseen_files = glob("/your_path/*.npz")

    print(f"共找到 seen 训练文件 {len(seen_files)} 个, unseen 验证文件 {len(unseen_files)} 个")

    # 处理 seen
    seen_metrics = []
    seen_gt_lists = {h: [] for h in HANDS}
    for f in tqdm.tqdm(seen_files):
        try:
            metrics, gt_seq = evaluate_file(f)
            seen_metrics.append(metrics)
            for h in HANDS:
                if h in gt_seq and isinstance(gt_seq[h], np.ndarray) and gt_seq[h].size > 0:
                    seen_gt_lists[h].append(gt_seq[h])
        except Exception as e:
            print(f"处理 seen 文件失败: {f}: {e}")
    seen_agg, seen_macro = aggregate(seen_metrics) if seen_metrics else ({}, {})

    # 处理 unseen
    unseen_metrics = []
    unseen_gt_lists = {h: [] for h in HANDS}
    for f in tqdm.tqdm(unseen_files):
        try:
            metrics, gt_seq = evaluate_file(f)
            unseen_metrics.append(metrics)
            for h in HANDS:
                if h in gt_seq and isinstance(gt_seq[h], np.ndarray) and gt_seq[h].size > 0:
                    unseen_gt_lists[h].append(gt_seq[h])
        except Exception as e:
            print(f"处理 unseen 文件失败: {f}: {e}")
    unseen_agg, unseen_macro = aggregate(unseen_metrics) if unseen_metrics else ({}, {})

    # 计算跨样本 gt_diversity 并注入 (两个 mode 都加, 方便展示)
    if seen_metrics:
        for h in HANDS:
            gtd = compute_gt_diversity_between_samples(seen_gt_lists[h])
            for mode in ["network", "optimization"]:
                seen_agg[h][mode]["gt_diversity"] = gtd
        for mode in ["network", "optimization"]:
            vals = [seen_agg[h][mode]["gt_diversity"] for h in HANDS if not np.isnan(seen_agg[h][mode]["gt_diversity"]) ]
            if vals:
                seen_macro[mode]["gt_diversity"] = float(np.mean(vals))
            else:
                seen_macro[mode]["gt_diversity"] = float('nan')
    if unseen_metrics:
        for h in HANDS:
            gtd = compute_gt_diversity_between_samples(unseen_gt_lists[h])
            for mode in ["network", "optimization"]:
                unseen_agg[h][mode]["gt_diversity"] = gtd
        for mode in ["network", "optimization"]:
            vals = [unseen_agg[h][mode]["gt_diversity"] for h in HANDS if not np.isnan(unseen_agg[h][mode]["gt_diversity"]) ]
            if vals:
                unseen_macro[mode]["gt_diversity"] = float(np.mean(vals))
            else:
                unseen_macro[mode]["gt_diversity"] = float('nan')

    if seen_metrics:
        print_results("Seen (Train)", seen_agg, seen_macro)
    if unseen_metrics:
        print_results("Unseen (Valid)", unseen_agg, unseen_macro)

if __name__ == "__main__":
    main()

