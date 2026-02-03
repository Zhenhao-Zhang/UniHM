from vizHandObj.hand_robot_viewer import RobotHandViewer
import numpy as np
from .utils import *


def _get_fingertip_links(robot):
    tips = [lnk for lnk in robot.get_links() if lnk.get_name().endswith("tip")]
    tips.sort(key=lambda l: l.get_name())
    return tips


def _classify_fingertips_svh(robot):
    """Classify SVH fingertips: return (thumb_index, other_indices).
    SVH tips typically named thtip, fftip, mftip, rftip, lftip.
    """
    tips = _get_fingertip_links(robot)
    names = [lnk.get_name().lower() for lnk in tips]
    thumb_idx = None
    for i, n in enumerate(names):
        if n == "thtip" or n.startswith("th") or "thumb" in n:
            thumb_idx = i
            break
    if thumb_idx is None:
        # fallback: first index if present
        thumb_idx = 0 if names else 0
    others = [i for i in range(len(names)) if i != thumb_idx]
    return thumb_idx, others


def _tip_positions_in_cam(viewer: RobotHandViewer, robot):
    cam_T_world = viewer.camera_pose.inv()
    tips = _get_fingertip_links(robot)
    P = []
    for tip in tips:
        lw = tip.get_entity_pose()
        lc = cam_T_world * lw
        P.append(lc.p)
    if not P:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(P, axis=0)


def _extract_joint_limits_safe(joint):
    lims = joint.get_limits()
    lower = upper = None
    if isinstance(lims, (tuple, list)) and len(lims) == 2:
        lower, upper = lims[0], lims[1]
    else:
        arr = np.array(lims)
        if arr.ndim == 2 and arr.shape[1] == 2:
            lower, upper = arr[0, 0], arr[0, 1]
        elif arr.ndim == 1 and arr.size >= 2:
            lower, upper = arr[0], arr[1]
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf
    lower = np.array(lower).reshape(-1)[0]
    upper = np.array(upper).reshape(-1)[0]
    return lower, upper


def _numerical_jacobian(viewer: RobotHandViewer, robot, q: np.ndarray, eps=1e-4):
    q = q.copy()
    robot.set_qpos(q)
    viewer.scene.update_render()
    p0 = _tip_positions_in_cam(viewer, robot).reshape(-1)
    m = len(robot.get_active_joints())
    J = np.zeros((p0.size, m), dtype=np.float64)
    joints = robot.get_active_joints()
    limits = [_extract_joint_limits_safe(j) for j in joints]
    lowers = np.array([l for l, _ in limits], dtype=np.float64)
    uppers = np.array([u for _, u in limits], dtype=np.float64)
    if p0.size == 0:
        return J, p0
    for j in range(m):
        qj = q.copy()
        qj[j] = np.clip(qj[j] + eps, lowers[j], uppers[j])
        robot.set_qpos(qj)
        viewer.scene.update_render()
        pj = _tip_positions_in_cam(viewer, robot).reshape(-1)
        J[:, j] = (pj - p0) / eps
    robot.set_qpos(q)
    viewer.scene.update_render()
    return J, p0


def optimize_svh_to_fixed_object(
    viewer: RobotHandViewer,
    objpose_seq: np.ndarray,
    grasped_obj_idx: int,
    obj_pc_local: np.ndarray,
    qpos_init: np.ndarray,
    iters_per_frame: int = 3,
    ik_lambda: float = 1e-3,
    dq_max: float = 0.05,
    start_frame: int = 0,
    contact_margin: float = 0.003,
    warm_frames: int = 8,
    reg_to_generated: float = 0.2,
    reg_to_prev: float = 0.5,
    blend_frames: int = 6,
    pre_contact_opt_frames: int = 6,
    pre_blend_frames: int = 6,
    pre_weight_power: float = 1.0,
    robot_index: int = 2,
):
    """Optimize SVH qpos to contact object using nearest points, with smooth transition.

    Mirror of Allegro optimizer but targets viewer.robots[robot_index] (default 2 for SVH).
    """
    objpose_np = np.asarray(objpose_seq)
    if objpose_np.ndim == 2:
        objpose_np = objpose_np[:, None, :]

    Tn = min(objpose_np.shape[0], qpos_init.shape[0])
    new_qpos = qpos_init.copy()

    robot = viewer.robots[robot_index]
    joints = robot.get_active_joints()
    limits = [_extract_joint_limits_safe(j) for j in joints]
    lowers = np.array([l for l, _ in limits], dtype=np.float64)
    uppers = np.array([u for _, u in limits], dtype=np.float64)

    if 0 <= start_frame < objpose_np.shape[0]:
        T_obj_fixed = posquat_to_T(objpose_np[start_frame, grasped_obj_idx, :])
    else:
        T_obj_fixed = posquat_to_T(objpose_np[0, grasped_obj_idx, :])

    t_pre_start = max(0, start_frame - max(0, pre_contact_opt_frames))

    for t in range(Tn):
        if t < t_pre_start:
            new_qpos[t] = qpos_init[t]
            continue

        pre_phase = (t < start_frame)
        use_fixed_obj = pre_phase or (t < start_frame + warm_frames)
        T_obj = T_obj_fixed if use_fixed_obj else posquat_to_T(objpose_np[t, grasped_obj_idx, :])

        q_gen = qpos_init[t].copy()
        if pre_phase:
            q = (new_qpos[t - 1] if t > t_pre_start else q_gen).copy()
        else:
            q = (new_qpos[t - 1] if t > start_frame else q_gen).copy()
        robot.set_qpos(q)
        viewer.scene.update_render()

        if pre_phase and pre_contact_opt_frames > 0:
            w_pre = ((t - t_pre_start + 1) / float(pre_contact_opt_frames))
            w_pre = float(np.clip(w_pre, 0.0, 1.0)) ** pre_weight_power
            margin_eff = contact_margin + (0.003 * (1.0 - w_pre))
        else:
            w_pre = 1.0
            margin_eff = contact_margin

        for _ in range(iters_per_frame):
            P = _tip_positions_in_cam(viewer, robot)
            if P.shape[0] == 0:
                break

            obj_cam = transform_points(T_obj, obj_pc_local)
            center = obj_cam.mean(axis=0) if obj_cam.size else np.zeros(3)
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

            # Encourage thumb opposition
            try:
                thumb_idx, other_idxs = _classify_fingertips_svh(robot)
                if P.shape[0] >= 2 and other_idxs:
                    p_thumb = P[thumb_idx]
                    vt = p_thumb - center
                    nt = np.linalg.norm(vt)
                    if nt > 1e-6:
                        ut = vt / nt
                        cos_target = -0.2
                        for oi in other_idxs:
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

            J, p_vec = _numerical_jacobian(viewer, robot, q)
            y = C_cam.reshape(-1)
            r = y - p_vec
            H = (w_pre * (J.T @ J)) + ik_lambda * np.eye(J.shape[1])
            g = (w_pre * (J.T @ r))

            if reg_to_generated > 0.0:
                H += reg_to_generated * np.eye(H.shape[0])
                g += reg_to_generated * (q_gen - q)

            reg_prev_eff = reg_to_prev * (2.0 if ((not pre_phase) and t == start_frame) else 1.0)
            if reg_prev_eff > 0.0 and (t > t_pre_start):
                q_prev = new_qpos[t - 1]
                H += reg_prev_eff * np.eye(H.shape[0])
                g += reg_prev_eff * (q_prev - q)

            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(H, g, rcond=None)[0]
            dq_max_eff = dq_max * (0.6 if ((not pre_phase) and t == start_frame) else 1.0)
            dq = np.clip(dq, -dq_max_eff, dq_max_eff)
            q = np.clip(q + dq, lowers, uppers)
            robot.set_qpos(q)
            viewer.scene.update_render()

        if pre_phase and pre_blend_frames > 0:
            alpha = (t - t_pre_start + 1) / float(pre_blend_frames)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            new_qpos[t] = (1.0 - alpha) * q_gen + alpha * q
        elif (not pre_phase) and blend_frames > 0 and t < start_frame + blend_frames:
            alpha = (t - start_frame + 1) / float(blend_frames)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            new_qpos[t] = (1.0 - alpha) * new_qpos[t - 1] + alpha * q
        else:
            new_qpos[t] = q

    return new_qpos
