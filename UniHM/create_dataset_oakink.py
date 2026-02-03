from vizHandObj.hand_robot_viewer import RobotHandViewer
from vizHandObj import ThreeJawGripperViewer
import numpy as np

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import sapien
import numpy as np
import torch
import tyro
from pytransform3d import rotations
from tqdm import tqdm

from vizHandObj.dataset import DexYCBVideoDataset
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
    ROBOT_NAME_MAP,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from vizHandObj.mano_layer import MANOLayer
from pytransform3d import transformations as pt



robot_dir = "/home/main/dex-ICLR/dex-retargeting/assets/robots/hands"
output_dir = "/home/main/data/oakink_robot_data"

import numpy as np
from glob import glob
import os 
RetargetingConfig.set_default_urdf_dir(robot_dir)
files = glob(f"/home/main/data/oakink/extracted_sequences/*.pkl")
viewer = RobotHandViewer(robot_names=[RobotName.allegro, 
                                      RobotName.shadow,
                                      RobotName.svh,
                                      RobotName.leap,
                                      RobotName.ability,
                                      RobotName.panda,
                                      ],
                          hand_type= HandType.right,
                          headless=True,
                          )
gripper_viewer = ThreeJawGripperViewer(urdf_path="/home/main/dex-ICLR/dex-retargeting/assets/robots/hands/dclaw_gripper/dclaw_gripper_glb.urdf", headless=True)

for i in tqdm(range(len(files))):
    try:
        data = np.load(files[i],allow_pickle=True)
        capture_name = data["seq_id"]
        # 替换mesh的跟路径
        mesh_root = "/home/main/data/oakink/obj"
        mesh_path = os.path.join(mesh_root, os.path.basename(data["object_mesh_path_list"][0]))
        data_load_obj = {
        "ycb_ids": [1],
        "object_mesh_file":mesh_path,
        "hand_shape":np.array(data["mano_shape"]),
        "extrinsics":np.identity(4)
        }
        viewer.load_object_hand(data_load_obj)
        data_ = {
        "ycb_ids":[1],
        "object_mesh_file":mesh_path,
        "object_pose":data["cam0_object_pose_seq_world"],
        "hand_shape":np.array(data["mano_shape"]),
        "extrinsics":np.identity(4),
        "hand_pose":np.expand_dims(np.array(data["mano_pose_seq"]),axis=1),
        "capture_name": data["seq_id"],
        "start_frame": 0
        }
        result = viewer.retargeting_only(data_)
        gripper_viewer.load_object_hand(data_load_obj)
        gripper_result = gripper_viewer.retargeting_only(data_)
        item = {
                    "robot_name": gripper_result["robot_name"],
                    "robot_qpos": gripper_result["robot_qpos"],
            }
        result["gripper"] = item
        ycb_ids_names = [1]
        result["ycb_ids_names"] = ycb_ids_names
        result["grasped_ycb_id"] = data["object_id"]
        result["grasped_ycb_name"] = data["object_id"]

        if result is not None:
            output_file = f"{output_dir}/{capture_name}"
            np.savez_compressed(
                f"{output_file}.npz",
                **result
            )
    except Exception as e:
        print(f"Failed to process {capture_name}: {e}")
    # viewer.render_dexycb_data(data_,
    #                       video_path="/home/main/dex-ICLR/UniHM/videos/hands.mp4",
    #                     #   precomputed_qpos = pre_pose,

    #                       )