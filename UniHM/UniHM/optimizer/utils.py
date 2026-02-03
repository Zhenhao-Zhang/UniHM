import numpy as np
import os
from pytransform3d import rotations as pr


def posquat_to_T(pq):
    """DexYCB [qx, qy, qz, qw, x, y, z] -> 4x4."""
    pq = np.asarray(pq)
    q_xyzw, pos = pq[0:4], pq[4:7]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    R = pr.matrix_from_quaternion(q_wxyz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def T_to_posquat(T):
    """4x4 -> DexYCB [qx, qy, qz, qw, x, y, z]."""
    pos = T[:3, 3]
    q_wxyz = pr.quaternion_from_matrix(T[:3, :3])
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    return np.concatenate([q_xyzw, pos], axis=0)

def transform_points(T, pts_local):
    R, t = T[:3, :3], T[:3, 3]
    return np.asarray(pts_local) @ R.T + t


def detect_object_motion_start(objpose_seq, grasped_obj_idx, trans_thresh=0.003, min_consecutive=3):
    """Detect first frame index when the grasped object starts moving.

    Args:
        objpose_seq: (T,K,7) or (T,7) posquat array (DexYCB order: qx,qy,qz,qw,x,y,z).
        grasped_obj_idx: index of the grasped object in K.
        trans_thresh: movement threshold in meters between consecutive frames.
        min_consecutive: require this many consecutive frames over threshold.
    Returns:
        start_frame (int) or None if not detected.
    """
    objpose_np = np.asarray(objpose_seq)
    if objpose_np.ndim == 2:
        objpose_np = objpose_np[:, None, :]
    if objpose_np.shape[0] < 2:
        return None
    pos = objpose_np[:, grasped_obj_idx, 4:7]
    diffs = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    count = 0
    for i, d in enumerate(diffs):
        if d > trans_thresh:
            count += 1
            if count >= min_consecutive:
                return max(1, i - min_consecutive + 2)
        else:
            count = 0
    return None

