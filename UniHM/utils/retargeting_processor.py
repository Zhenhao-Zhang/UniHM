from pathlib import Path
from typing import Dict, List, Optional

import sys
import numpy as np
import sapien
import torch
from pytransform3d import rotations
from pytransform3d import transformations as pt

from .mano_layer import MANOLayer

_ROOT = Path(__file__).resolve().parents[1]
_POS_RETARGET_DIR = _ROOT / "dex-retargeting" / "example" / "position_retargeting"
if str(_POS_RETARGET_DIR) not in sys.path:
    sys.path.append(str(_POS_RETARGET_DIR))

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    ROBOT_NAME_MAP,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting


class RetargetingProcessor:
    """Pure retargeting processor — no visualization, no rendering."""

    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
    ):
        sapien.render.set_log_level("error")
        scene = sapien.Scene()
        self.scene = scene

        self.robot_names = robot_names
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type
        self.sapien_joint_names: List[List[str]] = []
        self.robot_file_names: List[str] = []
        self.robots = []

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            self.robot_file_names.append(Path(config.urdf_path).stem)
            self.retargetings.append(retargeting)

            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            from tempfile import mkdtemp

            temp_dir = mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_path.name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.sapien_joint_names.append(sapien_joint_names)
            self.retarget2sapien.append(retarget2sapien)

        self.mano_layer: Optional[MANOLayer] = None
        self.camera_mat: Optional[np.ndarray] = None
        self.objects: List[sapien.Entity] = []

    def load_object_hand(self, data: Dict):
        self.setup_hand(data)
        
        for actor in self.objects:
            self.scene.remove_actor(actor)
        self.objects = []

        mesh_files = data.get("object_mesh_file", [])
        if not isinstance(mesh_files, list):
            mesh_files = [mesh_files]
        
        for mesh_path in mesh_files:
            builder = self.scene.create_actor_builder()
            if mesh_path:
                 builder.add_visual_from_file(str(mesh_path))
                 builder.add_multiple_convex_collisions_from_file(str(mesh_path))
            # Create a separate kinematic actor for each object mesh
            actor = builder.build_kinematic(name="object")
            self.objects.append(actor)

    def setup_hand(self, data: Dict):
        """Initialize MANO layer and camera transform from data.

        Required keys: ``hand_shape``, ``extrinsics``.
        """
        hand_shape = data["hand_shape"]
        extrinsic_mat = data["extrinsics"]
        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
        pose_vec = pt.pq_from_transform(extrinsic_mat)
        # In HandViewer, camera_pose is defined as the inverse (Camera -> World)
        # extrinsic_mat is World -> Camera
        self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()
        self.camera_mat = self.camera_pose.to_transformation_matrix()

    def _compute_joint_positions(self, hand_pose_frame: np.ndarray) -> Optional[np.ndarray]:
        """Compute MANO joint positions in world frame.

        Returns ``None`` when the hand pose is invalid (near-zero).
        """
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        _, joint = self.mano_layer(p, t)
        joint = joint.cpu().numpy()[0]
        joint = joint @ self.camera_mat[:3, :3].T + self.camera_mat[:3, 3]
        return np.ascontiguousarray(joint)

    def retarget(self, data: Dict) -> Dict:
        """Retarget a hand-pose trajectory to all loaded robots.

        Required keys: ``hand_pose``, ``capture_name``, ``object_pose``,
        ``extrinsics``, ``ycb_ids``, ``hand_shape``.
        Call :meth:`setup_hand` before this method.
        """
        hand_pose = data["hand_pose"]
        num_frame = hand_pose.shape[0]

        # Find first valid frame
        start_frame = 0
        for i in range(num_frame):
            joint = self._compute_joint_positions(hand_pose[i])
            if joint is not None:
                start_frame = i
                break

        # Warm-start retargeting optimizers
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )
        joint = self._compute_joint_positions(hand_pose_start)
        for retargeting in self.retargetings:
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        result = {
            "capture_name": data["capture_name"],
            "object_pose": data["object_pose"][start_frame:num_frame],
            "extrinsics": data["extrinsics"],
            "ycb_ids": data["ycb_ids"],
            "hand_shape": data["hand_shape"],
            "hand_pose": hand_pose[start_frame:num_frame, 0, :],
            "start_frame": start_frame,
        }
        if "object_mesh_file" in data:
            result["object_mesh_file"] = data["object_mesh_file"]

        # Retarget each frame
        robot_qpos_trajectories = [[] for _ in range(len(self.robot_names))]
        robot_index = {
            ROBOT_NAME_MAP[name]: i for i, name in enumerate(self.robot_names)
        }

        for i in range(start_frame, num_frame):
            joint = self._compute_joint_positions(hand_pose[i])
            for robotname, retargeting, retarget2sapien in zip(
                self.robot_names, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot_qpos_trajectories[
                    robot_index[ROBOT_NAME_MAP[robotname]]
                ].append(qpos)

        for robotname, qpos in zip(self.robot_names, robot_qpos_trajectories):
            item = {
                "robot_name": ROBOT_NAME_MAP[robotname],
                "robot_qpos": np.array(qpos),
            }
            result[ROBOT_NAME_MAP[robotname]] = item

        return result
