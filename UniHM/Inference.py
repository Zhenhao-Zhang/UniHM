from vizHandObj.hand_robot_viewer import RobotHandViewer
import numpy as np
import os
import argparse
from glob import glob
from pytransform3d import rotations as pr
from UniHM.optimizer import optimize_allegro_to_fixed_object,detect_object_motion_start, optimize_mano_to_fixed_object
from UniHM.optimizer import optimize_shadow_to_fixed_object
from UniHM.optimizer import optimize_svh_to_fixed_object
from UniHM.dataset import load_dataset_squential
from dex_retargeting.retargeting_config import RetargetingConfig
RetargetingConfig.set_default_urdf_dir("/your_path")
from dex_retargeting.constants import RobotName, HandType
from UniHM.SFT.utils import build_seq_dataloaders, build_seq_dataloaders_list  # added build_seq_dataloaders_list
import json
from tqdm import tqdm
import torch
import random

def _base_name(k: str):
    if k == "mano":
        return "mano"
    if k == "hand_pose":
        return "mano"
    if k.endswith("_hand_qpos"):
        return k.replace("_hand_qpos", "")
    if k.endswith("_gripper_qpos"):
        return k.replace("_gripper_qpos", "")
    return k

OPT_HAND_SET = {"mano", "allegro", "shadow", "svh"}


def export_dataset(args):

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from UniHM.SFT import build_qwen_vqvae_aligner
    vqvae_kwargs = dict(
        in_dim=1,
        h_dim=128,
        res_h_dim=128,
        n_res_layers=2,
        n_embeddings=8192,
        embedding_dim=512,
        beta=0.25,
        num_decoders=6,
        decoder_out_channels=[22, 30, 26, 22, 16, 8],
        use_mlp=False,
        input_length=51,
    )
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    qwen_dtype = dtype_map.get(args.qwen_dtype, torch.bfloat16)
    model = build_qwen_vqvae_aligner(
        vqvae_ckpt_path=args.vqvae_ckpt,
        vqvae_kwargs=vqvae_kwargs,
        qwen_model_name_or_path=args.qwen_model,
        device=device,
        freeze_vqvae=True,
        n_object_tokens=0,
        qwen_dtype=qwen_dtype,
    )
    ckpt = torch.load(args.sft_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    present_robot_keys = ckpt["present_robot_keys"]

    # ------------- 新增：支持 train.txt / valid.txt 列表 -------------
    if args.train_list and args.valid_list and os.path.exists(args.train_list) and os.path.exists(args.valid_list):
        def _read_list(path):
            with open(path, 'r') as f:
                lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            return lines
        train_files = _read_list(args.train_list)
        valid_files = _read_list(args.valid_list)
        train_files = [p for p in train_files if os.path.exists(p)]
        valid_files = [p for p in valid_files if os.path.exists(p)]
        if len(train_files) == 0:
            raise RuntimeError(f"No training files found in list {args.train_list}")
        if len(valid_files) == 0:
            raise RuntimeError(f"No validation files found in list {args.valid_list}")
        train_loader, val_loader = build_seq_dataloaders_list(train_files, valid_files, batch_size=1, num_workers=0)
    else:
        print("using glob")
        # train_loader, val_loader = build_seq_dataloaders(args.data_glob, batch_size=1, num_workers=0)
        # train_files = list(train_loader.dataset.files)
        # val_files = list(val_loader.dataset.files)
    # -------------------------------------------------------------

    out_root_train = "/your_path/train"
    out_root_valid = "/your_path/valid"
    os.makedirs(out_root_train, exist_ok=True)
    os.makedirs(out_root_valid, exist_ok=True)

    def process_files(file_list, split_name, out_dir):
        for idx, fp in enumerate(tqdm(file_list, desc=f"{split_name} export")):
            try:
                ds = load_dataset_squential(fp)
            except Exception as e:
                print(f"[WARN] Failed loading {fp}: {e}")
                continue
            extrinsics = ds["extrinsics"]
            hand_shape = ds["hand_shape"].cpu().numpy() if hasattr(ds["hand_shape"], "cpu") else ds["hand_shape"]
            ycb_ids = ds["ycb_ids"]
            objmesh = ds["object_mesh_file"]
            objpose = ds["object_pose"]
            grasped_obj_idx = ds["grasped_obj_idx"]
            obj_pc_local = ds["grasped_obj_point3d"].cpu().numpy() if hasattr(ds["grasped_obj_point3d"], "cpu") else ds["grasped_obj_point3d"]

            mano_pose = ds["hand_pose"].to(torch.float32).to(device).unsqueeze(0)
            pointcloud = ds["grasped_obj_point3d"].to(torch.float32).to(device).unsqueeze(0)
            obj_pose_seq = torch.tensor(ds["grasped_obj_pose"]).to(torch.float32).to(device).unsqueeze(0)
            text = f"grasp object id {ds.get('grasped_with_obj_id', '')}"
            tok = model.tokenizer([text], padding=True, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}
            # 对objpose添加10%噪声
            noise = torch.randn_like(obj_pose_seq) * torch.max(obj_pose_seq) * 2
            objpose_noise =  obj_pose_seq + noise

            mano_mask = torch.zeros((mano_pose.shape[0], mano_pose.shape[1]), device=device)
            mano_mask[:, : args.mano_prefix_frames] = 1.0
            # -------------------------
            # First generation (generation)
            # -------------------------
            with torch.no_grad():
                out = model(
                    mano_pose=mano_pose,
                    object_pointcloud=pointcloud,
                    object_pose_seq=objpose_noise,
                    text_inputs=tok,
                    decoder_branch=args.decoder_branch,
                    text_position=args.text_position,
                    mano_mask=mano_mask,
                )
                q_hidden = out["qwen_hidden"].squeeze(0)
                mano_vq_hat = model.from_qwen(q_hidden)
                generation = {}
                for i, key in enumerate(present_robot_keys):
                    pred_bt = model.vqvae.decode(mano_vq_hat.to(torch.float32), branch=i).squeeze(-1)
                    generation[_base_name(key)] = pred_bt.cpu().numpy()
                if hasattr(model.vqvae, 'mano_decoder') and model.vqvae.mano_decoder is not None:
                    generation['mano'] = model.vqvae.mano_decoder(mano_vq_hat.to(torch.float32)).squeeze(-1).cpu().numpy()

            # -------------------------
            # Second generation pass (generation2) - same prompt to measure stochastic differences
            # -------------------------
            with torch.no_grad():
                out2 = model(
                    mano_pose=mano_pose,
                    object_pointcloud=pointcloud,
                    object_pose_seq=obj_pose_seq,
                    text_inputs=tok,  # same text
                    decoder_branch=args.decoder_branch,
                    text_position=args.text_position,
                    mano_mask=mano_mask,
                )
                q_hidden2 = out2["qwen_hidden"].squeeze(0)
                mano_vq_hat2 = model.from_qwen(q_hidden2)
                generation2 = {}
                for i, key in enumerate(present_robot_keys):
                    pred_bt2 = model.vqvae.decode(mano_vq_hat2.to(torch.float32), branch=i).squeeze(-1)
                    generation2[_base_name(key)] = pred_bt2.cpu().numpy()
                if hasattr(model.vqvae, 'mano_decoder') and model.vqvae.mano_decoder is not None:
                    generation2['mano'] = model.vqvae.mano_decoder(mano_vq_hat2.to(torch.float32)).squeeze(-1).cpu().numpy()

            # -------------------------
            # Similar text generation (generation_sim) - modified prompt to measure semantic robustness
            # -------------------------
            sim_text = f"use the object, which id is {ds.get('grasped_with_obj_id', '')}"
            tok_sim = model.tokenizer([sim_text], padding=True, return_tensors="pt")
            tok_sim = {k: v.to(device) for k, v in tok_sim.items()}
            with torch.no_grad():
                out_sim = model(
                    mano_pose=mano_pose,
                    object_pointcloud=pointcloud,
                    object_pose_seq=obj_pose_seq,
                    text_inputs=tok_sim,
                    decoder_branch=args.decoder_branch,
                    text_position=args.text_position,
                    mano_mask=mano_mask,
                )
                q_hidden_sim = out_sim["qwen_hidden"].squeeze(0)
                mano_vq_hat_sim = model.from_qwen(q_hidden_sim)
                generation_sim = {}
                for i, key in enumerate(present_robot_keys):
                    pred_bt_sim = model.vqvae.decode(mano_vq_hat_sim.to(torch.float32), branch=i).squeeze(-1)
                    generation_sim[_base_name(key)] = pred_bt_sim.cpu().numpy()
                if hasattr(model.vqvae, 'mano_decoder') and model.vqvae.mano_decoder is not None:
                    generation_sim['mano'] = model.vqvae.mano_decoder(mano_vq_hat_sim.to(torch.float32)).squeeze(-1).cpu().numpy()
            # Raw data (all available hand poses)
            raw_block = {}
            raw_block['mano'] = mano_pose.squeeze(0).cpu().numpy()
            for k, v in ds.items():
                if k.endswith('_qpos'):
                    raw_block[_base_name(k)] = v.cpu().numpy() if hasattr(v, 'cpu') else v

            # Optimization (only selected hands if present in generation)
            optimization = {}
            # Setup viewer for optimization
            viewer = RobotHandViewer(
                robot_names=[RobotName.allegro, RobotName.shadow, RobotName.svh],
                hand_type=HandType.right,
                headless=True,
            )
            objmesh = [objmesh.tolist()]
            viewer.load_object_hand({
                "ycb_ids": ycb_ids,
                "object_mesh_file": objmesh,
                "hand_shape": hand_shape,
                "extrinsics": extrinsics,
            })
            objpose_aligned = objpose
            contact_start = detect_object_motion_start(
                objpose_aligned,
                grasped_obj_idx,
                trans_thresh=args.motion_trans_thresh,
                min_consecutive=args.motion_min_consecutive,
            )
            if contact_start is None:
                contact_start = args.contact_fallback_frames
            # MANO optimization
            # if 'mano' in generation:
            #     try:
            #         mano_opt = optimize_mano_to_fixed_object(
            #             viewer,
            #             objpose_aligned,
            #             grasped_obj_idx,
            #             obj_pc_local,
            #             generation['mano'],
            #             iters_per_frame=args.iters_per_frame,
            #             ik_lambda=args.ik_lambda,
            #             dq_max=args.dq_max,
            #             start_frame=contact_start,
            #             contact_margin=args.contact_margin,
            #             warm_frames=args.warm_frames,
            #             reg_to_generated=args.reg_to_generated,
            #             reg_to_prev=args.reg_to_prev,
            #             blend_frames=args.blend_frames,
            #             pre_contact_opt_frames=args.pre_contact_opt_frames,
            #             pre_blend_frames=args.pre_blend_frames,
            #             pre_weight_power=args.pre_weight_power,
            #         )
            #         optimization['mano'] = mano_opt
            #     except Exception as e:
            #         print(f"[WARN] MANO optimization failed {fp}: {e}")
            # Allegro
            if 'allegro' in generation:
                try:
                    allegro_opt = optimize_allegro_to_fixed_object(
                        viewer,
                        objpose_aligned,
                        grasped_obj_idx,
                        obj_pc_local,
                        generation['allegro'],
                        iters_per_frame=args.iters_per_frame,
                        ik_lambda=args.ik_lambda,
                        dq_max=args.dq_max,
                        start_frame=contact_start,
                        contact_margin=args.contact_margin,
                        warm_frames=args.warm_frames,
                        reg_to_generated=args.reg_to_generated,
                        reg_to_prev=args.reg_to_prev,
                        blend_frames=args.blend_frames,
                        pre_contact_opt_frames=args.pre_contact_opt_frames,
                        pre_blend_frames=args.pre_blend_frames,
                        pre_weight_power=args.pre_weight_power,
                    )
                    optimization['allegro'] = allegro_opt
                except Exception as e:
                    print(f"[WARN] Allegro optimization failed {fp}: {e}")
            # Shadow (robot_index=1)
            if 'shadow' in generation:
                try:
                    shadow_opt = optimize_shadow_to_fixed_object(
                        viewer,
                        objpose_aligned,
                        grasped_obj_idx,
                        obj_pc_local,
                        generation['shadow'],
                        iters_per_frame=args.iters_per_frame,
                        ik_lambda=args.ik_lambda,
                        dq_max=args.dq_max,
                        start_frame=contact_start,
                        contact_margin=args.contact_margin,
                        warm_frames=args.warm_frames,
                        reg_to_generated=args.reg_to_generated,
                        reg_to_prev=args.reg_to_prev,
                        blend_frames=args.blend_frames,
                        pre_contact_opt_frames=args.pre_contact_opt_frames,
                        pre_blend_frames=args.pre_blend_frames,
                        pre_weight_power=args.pre_weight_power,
                        robot_index=1,
                    )
                    optimization['shadow'] = shadow_opt
                except Exception as e:
                    print(f"[WARN] Shadow optimization failed {fp}: {e}")
            # SVH (robot_index=2) - keys might be 'svh' derived from 'svh_hand_qpos' or 'schunk_svh_hand_qpos'
            if 'svh' in generation:
                try:
                    svh_opt = optimize_svh_to_fixed_object(
                        viewer,
                        objpose_aligned,
                        grasped_obj_idx,
                        obj_pc_local,
                        generation['svh'],
                        iters_per_frame=args.iters_per_frame,
                        ik_lambda=args.ik_lambda,
                        dq_max=args.dq_max,
                        start_frame=contact_start,
                        contact_margin=args.contact_margin,
                        warm_frames=args.warm_frames,
                        reg_to_generated=args.reg_to_generated,
                        reg_to_prev=args.reg_to_prev,
                        blend_frames=args.blend_frames,
                        pre_contact_opt_frames=args.pre_contact_opt_frames,
                        pre_blend_frames=args.pre_blend_frames,
                        pre_weight_power=args.pre_weight_power,
                        robot_index=2,
                    )
                    optimization['svh'] = svh_opt
                except Exception as e:
                    print(f"[WARN] SVH optimization failed {fp}: {e}")

            sample = {
                "idx": idx,
                "hand_shape": hand_shape.tolist() if hasattr(hand_shape, 'tolist') else hand_shape,
                "ycb_ids": ycb_ids.tolist() if hasattr(ycb_ids, 'tolist') else ycb_ids,
                "extrinsics": extrinsics.tolist() if hasattr(extrinsics, 'tolist') else extrinsics,
                "object_mesh_file": list(objmesh),
                "object_pose": objpose.tolist() if hasattr(objpose, 'tolist') else objpose,
                "background_with_obj_names": ds.get('background_with_obj_names', ''),
                "background_with_obj_ids": ds.get('background_with_obj_ids', ''),
                "grasped_with_obj_name": ds.get('grasped_with_obj_name', ''),
                "grasped_with_obj_id": ds.get('grasped_with_obj_id', ''),
                "grasped_obj_pose": ds.get('grasped_obj_pose').tolist() if hasattr(ds.get('grasped_obj_pose'), 'tolist') else ds.get('grasped_obj_pose'),
                "grasped_obj_idx": int(ds.get('grasped_obj_idx', -1)),
                "raw": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in raw_block.items()},
                "generation": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in generation.items()},
                "generation2": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in generation2.items()},
                "generation_sim": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in generation_sim.items()},
                "optimization": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in optimization.items()},
            }
            out_name = os.path.splitext(os.path.basename(fp))[0] + '.npz'
            out_path = os.path.join(out_dir, out_name)
            try:
                np.savez_compressed(out_path, data=sample)
            except Exception:
                # fallback without compression
                np.savez(out_path, data=sample)
        print(f"{split_name} saved to {out_dir}")

    process_files(train_files, 'train', out_root_train)
    process_files(valid_files, 'valid', out_root_valid)
    print("导出完成 (per-file npz)")


def main():
    parser = argparse.ArgumentParser(description="Generate / optimize and optionally export dataset")
    parser.add_argument("--data_glob", default="/home/your_path/data/robot_data/*.npz")
    # 新增：文件列表方式（若提供则优先生效）
    parser.add_argument("--train-list", type=str, default="/home/your_path/UniHM/UniHM/train_oak.txt", help="Path to train.txt (one npz path per line)")
    parser.add_argument("--valid-list", type=str, default="/home/your_path/UniHM/UniHM/valid_oak.txt", help="Path to valid.txt (one npz path per line)")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="e.g., cuda, cuda:0, cuda:1, or cpu")
    parser.add_argument("--vqvae_ckpt", default="/home/your_path/UniHM/UniHM/UniHM/ckpt/memd/conv1d/memd_conv1d.pth")
    parser.add_argument("--sft_ckpt", default="/home/your_path/UniHM/UniHM/sft_best_oak.pth")
    parser.add_argument("--video_path", default="/home/your_path/UniHM/UniHM/videos/hands_gen.mp4")
    # 模型与解码配置
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-0.6B", help="Qwen 模型名称或本地路径")
    parser.add_argument("--qwen_dtype", default="fp32", choices=["bf16", "fp16", "fp32"], help="Qwen 推理精度：bf16/fp16/fp32")
    parser.add_argument("--decoder_branch", type=int, default=2, help="VQVAE 解码分支索引（0=Allegro 分支）")
    parser.add_argument("--text_position", default="prefix", help="文本与时序结合方式，默认 prefix")
    # 生成前缀与接触检测
    parser.add_argument("--mano_prefix_frames", type=int, default=5, help="Qwen 生成时使用的前缀监督帧数")
    parser.add_argument("--motion_trans_thresh", type=float, default=0.003, help="物体开始移动检测的平移阈值(米)")
    parser.add_argument("--motion_min_consecutive", type=int, default=3, help="达到阈值的最少连续帧数")
    parser.add_argument("--contact_fallback_frames", type=int, default=5, help="未检测到运动时，接触起点的备用前缀帧数")
    # 优化与平滑相关参数
    parser.add_argument("--iters_per_frame", type=int, default=10, help="每帧高斯牛顿优化迭代次数")
    parser.add_argument("--ik_lambda", type=float, default=1e-3, help="阻尼项 λ，用于稳定求解 (H+λI)")
    parser.add_argument("--dq_max", type=float, default=0.03, help="单次迭代的关节步长上限")
    parser.add_argument("--contact_margin", type=float, default=0.01, help="指尖目标点向外的安全边距(米)，减小穿模")
    parser.add_argument("--warm_frames", type=int, default=8, help="接触后固定物体位姿的暖启动帧数")
    parser.add_argument("--reg_to_generated", type=float, default=0.7, help="正则到生成轨迹的权重，抑制跳变并保持风格")
    parser.add_argument("--reg_to_prev", type=float, default=0.5, help="正则到前一优化帧的权重（速度平滑）")
    parser.add_argument("--blend_frames", type=int, default=20, help="接触后与优化解混合的平滑帧数")
    parser.add_argument("--pre_contact_opt_frames", type=int, default=8, help="接触前的优化靠拢帧数（固定物体位姿）")
    parser.add_argument("--pre_blend_frames", type=int, default=8, help="接触前与生成轨迹混合的平滑帧数")
    parser.add_argument(
        "--pre_weight_power",
        type=float,
        default=1.0,
        help="接触前残差权重递增的幂次（>1 更慢启动更稳）",
    )
    # Export options
    parser.add_argument("--log_interval", type=int, default=1, help="导出进度打印间隔")

    args = parser.parse_args()

    export_dataset(args)

    # 可扩展: 其它 pipeline

if __name__ == "__main__":
    main()