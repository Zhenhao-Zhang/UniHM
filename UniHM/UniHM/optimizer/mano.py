import numpy as np
from typing import Tuple
from vizHandObj.hand_robot_viewer import RobotHandViewer
from .utils import posquat_to_T, transform_points
# NEW: import torch for batched MANO forward / autograd
import torch

# MANO specifics
# We assume 21 MANO joints ordered as:
# 0: wrist,
# Index finger: 1-4 (4 is tip)
# Middle finger: 5-8 (8 is tip)
# Ring finger: 9-12 (12 is tip)
# Pinky finger: 13-16 (16 is tip)
# Thumb: 17-20 (20 is tip)
# Hence fingertip indices are [4, 8, 12, 16, 20] and we will order them as
# [index, middle, ring, pinky, thumb]
_MANO_TIP_IDXS = [4, 8, 12, 16, 20]
_THUMB_TIP_LOCAL_IDX = 4  # index within the ordered tips list


def _mano_tip_positions_in_cam(viewer: RobotHandViewer, mano_vec51: np.ndarray) -> np.ndarray:
    """Compute MANO fingertip positions (camera frame) for a given 51D vector.

    Args:
        viewer: active viewer with MANO layer initialized via load_object_hand.
        mano_vec51: shape (51,) array, 48 pose + 3 trans.
    Returns:
        (5, 3) fingertip positions in camera frame ordered as
        [index, middle, ring, pinky, thumb]. If geometry is invalid, returns (0,3).
    """
    hp = mano_vec51.reshape(1, 51)
    vertex, joint = viewer._compute_hand_geometry(hp, use_camera_frame=True)
    if vertex is None or joint is None:
        return np.zeros((0, 3), dtype=np.float64)
    tips = np.asarray(joint)[_MANO_TIP_IDXS, :].astype(np.float64)
    return tips


def _numerical_jacobian_mano(
    viewer: RobotHandViewer,
    x: np.ndarray,
    eps_pose: float = 1e-3,
    eps_trans: float = 1e-3,
    use_batch: bool = True,
    use_autograd: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian of fingertip positions wrt 51D MANO vector.

    Priority:
      1) Autograd (pose) + analytic translation (fastest, stable) if use_autograd.
      2) Batched finite difference for pose (single forward) + analytic translation if use_batch.
      3) Per-dof finite difference fallback.
    Returns (J (15,51), p0 (15,)). Empty arrays if geometry invalid.
    """
    mano_layer = getattr(viewer, 'mano_layer', None)
    if mano_layer is None or x.size != 51:
        use_autograd = False
        use_batch = False

    pose_np = x[:48].astype(np.float32)
    trans_np = x[48:51].astype(np.float32)

    # -------- Autograd path --------
    if use_autograd and mano_layer is not None:
        try:
            # Determine device
            try:
                device = next(mano_layer.parameters()).device
            except StopIteration:
                # If MANO layer has no parameters (unlikely), fallback to CPU
                device = torch.device('cpu')
            pose_t = torch.tensor(pose_np, device=device, requires_grad=True)
            trans_t = torch.tensor(trans_np, device=device)  # translation: no grad needed (analytic)

            def _forward_pose(p):
                v, j = mano_layer(p.unsqueeze(0), trans_t.unsqueeze(0))  # (1,V,3),(1,21,3)
                tips = j[0, _MANO_TIP_IDXS, :]  # (5,3)
                return tips.reshape(-1)  # (15,)

            with torch.enable_grad():
                tips0 = _forward_pose(pose_t)
                # Jacobian wrt pose (15,48)
                J_pose = torch.autograd.functional.jacobian(_forward_pose, pose_t, create_graph=False)
            p0 = tips0.detach().cpu().numpy()
            J_pose_np = J_pose.detach().cpu().numpy()  # (15,48)
            # Assemble full J
            J = np.zeros((15, 51), dtype=np.float64)
            J[:, :48] = J_pose_np
            # Translation analytic: identity replicated per tip
            for k in range(5):
                rs = slice(k * 3, k * 3 + 3)
                J[rs, 48:51] = np.eye(3)
            return J, p0
        except Exception:
            # Fallback to batch / finite difference
            pass

    # -------- Batched finite difference for pose only --------
    if use_batch and mano_layer is not None:
        try:
            # Build batch: baseline + 48 pose perturbations
            B = 49
            pose_batch = np.tile(pose_np, (B, 1))
            for i in range(48):
                pose_batch[i + 1, i] += eps_pose
            trans_batch = np.tile(trans_np, (B, 1))
            p_tensor = torch.from_numpy(pose_batch)
            t_tensor = torch.from_numpy(trans_batch)
            with torch.no_grad():
                _, joint_batch = mano_layer(p_tensor, t_tensor)  # (B,21,3)
            joints_np = joint_batch.cpu().numpy()
            tips_all = joints_np[:, _MANO_TIP_IDXS, :]  # (B,5,3)
            if tips_all.shape[0] == 0:
                return np.zeros((0, x.size), dtype=np.float64), np.zeros((0,), dtype=np.float64)
            tips0 = tips_all[0]
            p0 = tips0.reshape(-1)
            J = np.zeros((15, 51), dtype=np.float64)
            for i in range(48):
                diff = (tips_all[i + 1] - tips0) / eps_pose
                J[:, i] = diff.reshape(-1)
            for k in range(5):
                rs = slice(k * 3, k * 3 + 3)
                J[rs, 48:51] = np.eye(3)
            return J, p0
        except Exception:
            pass

    # -------- Fallback: full finite difference (including translation) --------
    P = _mano_tip_positions_in_cam(viewer, x)
    if P.shape[0] == 0:
        return np.zeros((0, x.size), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    p0 = P.reshape(-1)
    m = x.size
    J = np.zeros((p0.size, m), dtype=np.float64)
    for j in range(m):
        xj = x.copy()
        step = eps_pose if j < 48 else eps_trans
        xj[j] = xj[j] + step
        Pj = _mano_tip_positions_in_cam(viewer, xj)
        pj = Pj.reshape(-1)
        J[:, j] = (pj - p0) / step
    return J, p0


def optimize_mano_to_fixed_object(
    viewer: RobotHandViewer,
    objpose_seq: np.ndarray,
    grasped_obj_idx: int,
    obj_pc_local: np.ndarray,
    mano_init: np.ndarray,
    iters_per_frame: int = 3,
    ik_lambda: float = 1e-3,
    dq_max: float = 0.05,
    start_frame: int = 0,
    contact_margin: float = 0.0035,
    warm_frames: int = 8,
    reg_to_generated: float = 0.2,
    reg_to_prev: float = 0.5,
    blend_frames: int = 6,
    pre_contact_opt_frames: int = 6,
    pre_blend_frames: int = 6,
    pre_weight_power: float = 1.0,
    eps_pose: float = 1e-3,
    eps_trans: float = 1e-3,
    use_batch_jac: bool = True,
    use_autograd_jac: bool = True,
    early_stop_dx: float = 5e-4,
    early_stop_res: float = 1e-3,
) -> np.ndarray:
    """Optimized MANO trajectory refinement.

    Performance improvements over original:
    - Batched Jacobian (single MANO forward) + analytic translation columns.
    - Cache object point cloud transform per frame (not per iteration).
    - Early stopping inside per-frame iterations if update or residual small.
    Parameters eps_pose, eps_trans, use_batch_jac to control Jacobian strategy.
    """
    objpose_np = np.asarray(objpose_seq)
    if objpose_np.ndim == 2:
        objpose_np = objpose_np[:, None, :]

    Tn = min(objpose_np.shape[0], mano_init.shape[0])
    new_mano = mano_init.copy()

    if 0 <= start_frame < objpose_np.shape[0]:
        T_obj_fixed = posquat_to_T(objpose_np[start_frame, grasped_obj_idx, :])
    else:
        T_obj_fixed = posquat_to_T(objpose_np[0, grasped_obj_idx, :])

    t_pre_start = max(0, start_frame - max(0, pre_contact_opt_frames))

    for t in range(Tn):
        if t < t_pre_start:
            new_mano[t] = mano_init[t]
            continue
        pre_phase = (t < start_frame)
        use_fixed_obj = pre_phase or (t < start_frame + warm_frames)
        T_obj = T_obj_fixed if use_fixed_obj else posquat_to_T(objpose_np[t, grasped_obj_idx, :])
        obj_cam = transform_points(T_obj, obj_pc_local)
        center = obj_cam.mean(axis=0) if obj_cam.size else np.zeros(3)
        x_gen = mano_init[t].copy()
        if pre_phase:
            x = (new_mano[t - 1] if t > t_pre_start else x_gen).copy()
        else:
            x = (new_mano[t - 1] if t > start_frame else x_gen).copy()
        if pre_phase and pre_contact_opt_frames > 0:
            w_pre = ((t - t_pre_start + 1) / float(pre_contact_opt_frames))
            w_pre = float(np.clip(w_pre, 0.0, 1.0)) ** pre_weight_power
            margin_eff = contact_margin + (0.003 * (1.0 - w_pre))
        else:
            w_pre = 1.0
            margin_eff = contact_margin
        for _ in range(iters_per_frame):
            P = _mano_tip_positions_in_cam(viewer, x)
            if P.shape[0] == 0:
                break
            C_cam = []
            for i in range(P.shape[0]):
                d2 = np.sum((obj_cam - P[i][None, :]) ** 2, axis=1)
                j = int(np.argmin(d2))
                c = obj_cam[j].copy()
                u = c - center
                nu = np.linalg.norm(u)
                di = np.linalg.norm(c - P[i])
                if nu > 1e-8 and di < margin_eff:
                    c = c + (margin_eff - di) * (u / nu)
                C_cam.append(c)
            C_cam = np.vstack(C_cam)
            try:
                thumb_idx = _THUMB_TIP_LOCAL_IDX
                if P.shape[0] >= 2:
                    p_thumb = P[thumb_idx]
                    vt = p_thumb - center
                    nt = np.linalg.norm(vt)
                    if nt > 1e-6:
                        ut = vt / nt
                        cos_target = -0.2
                        for oi in range(P.shape[0]):
                            if oi == thumb_idx:
                                continue
                            vo = C_cam[oi] - center
                            no = np.linalg.norm(vo)
                            if no < 1e-6:
                                continue
                            uo = vo / no
                            cos_val = float(np.clip(np.dot(ut, uo), -1.0, 1.0))
                            if cos_val > cos_target:
                                step = 0.01 * (cos_val - cos_target)
                                step = float(np.clip(step, 0.0, 0.003))
                                C_cam[oi, :] = C_cam[oi, :] - step * ut
            except Exception:
                pass
            J, p_vec = _numerical_jacobian_mano(
                viewer,
                x,
                eps_pose=eps_pose,
                eps_trans=eps_trans,
                use_batch=use_batch_jac,
                use_autograd=use_autograd_jac,
            )
            if J.shape[0] == 0:
                break
            y = C_cam.reshape(-1)
            r = y - p_vec
            res_norm = float(np.linalg.norm(r))
            H = (w_pre * (J.T @ J)) + ik_lambda * np.eye(J.shape[1])
            g = (w_pre * (J.T @ r))
            if reg_to_generated > 0.0:
                H += reg_to_generated * np.eye(H.shape[0])
                g += reg_to_generated * (x_gen - x)
            reg_prev_eff = reg_to_prev * (2.0 if ((not pre_phase) and t == start_frame) else 1.0)
            if reg_prev_eff > 0.0 and (t > t_pre_start):
                x_prev = new_mano[t - 1]
                H += reg_prev_eff * np.eye(H.shape[0])
                g += reg_prev_eff * (x_prev - x)
            try:
                dx = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, g, rcond=None)[0]
            dq_max_eff = dq_max * (0.6 if ((not pre_phase) and t == start_frame) else 1.0)
            dx = np.clip(dx, -dq_max_eff, dq_max_eff)
            x = x + dx
            if np.max(np.abs(dx)) < early_stop_dx or res_norm < early_stop_res:
                break
        if pre_phase and pre_blend_frames > 0:
            alpha = (t - t_pre_start + 1) / float(pre_blend_frames)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            new_mano[t] = (1.0 - alpha) * x_gen + alpha * x
        elif (not pre_phase) and blend_frames > 0 and t < start_frame + blend_frames:
            alpha = (t - start_frame + 1) / float(blend_frames)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            new_mano[t] = (1.0 - alpha) * new_mano[t - 1] + alpha * x
        else:
            new_mano[t] = x
    return new_mano
