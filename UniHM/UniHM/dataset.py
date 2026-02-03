import numpy as np
import torch
from glob import glob
import os 
import trimesh


def load_dataset_squential(file):
    data = np.load(file,allow_pickle=True)

    extrinsics = data["extrinsics"]
    hand_shape = data["hand_shape"]

    hand_pose = data["hand_pose"]

    object_mesh_file = data["object_mesh_file"]
    object_pose = data["object_pose"]

    allegro_hand_qpos = dict(data["allegro_hand"].tolist())["robot_qpos"]
    shadow_hand_qpos = dict(data["shadow_hand"].tolist())["robot_qpos"]
    schunk_svh_hand_qpos = dict(data["schunk_svh_hand"].tolist())["robot_qpos"]
    leap_hand_qpos = dict(data["leap_hand"].tolist())["robot_qpos"]
    ability_hand_qpos = dict(data["ability_hand"].tolist())["robot_qpos"]
    panda_gripper_qpos = dict(data["panda_gripper"].tolist())["robot_qpos"]
    gripper_qpos = dict(data["panda_gripper"].tolist())["robot_qpos"]

    ycb_ids = data["ycb_ids"]
    ycb_ids_names = data["ycb_ids_names"]
    grasped_ycb_id = data["grasped_ycb_id"]
    grasped_ycb_name = data["grasped_ycb_name"]

    try:
        background_with_obj_names = ", ".join(ycb_ids_names)+" list on the table"
        background_with_obj_ids = "objects" + ", ".join([str(i) for i in ycb_ids.tolist()])+" list on the table"
    except:
        background_with_obj_names = "background with objects"
        background_with_obj_ids = "background with objects"
    grasped_with_obj_name = "grasp "+ grasped_ycb_name
    grasped_with_obj_id = "grasp " + str(grasped_ycb_id)

    try:
        grasped_obj_idx = ycb_ids.tolist().index(grasped_ycb_id)
        grasped_obj_pose = object_pose[:,grasped_obj_idx]
    except:
        grasped_obj_idx = 0
        grasped_obj_pose = object_pose[:,0]
    try:
        grasped_obj_xyz = [os.path.join(os.path.dirname(obj),"points.xyz") for obj in object_mesh_file][grasped_obj_idx]
        grasped_obj_point3d = np.loadtxt(grasped_obj_xyz)
    except:
        grasped_obj_point3d = np.loadtxt(object_mesh_file.tolist())
        # mesh = trimesh.load(object_mesh_file.tolist(), process=False)
        # grasped_obj_point3d = mesh.vertices.view(np.ndarray)



    result = {
        "hand_shape": torch.tensor(hand_shape),
        "hand_pose": torch.tensor(hand_pose),
        "ycb_ids": ycb_ids,
        "extrinsics": extrinsics,
        "object_mesh_file": object_mesh_file,
        "object_pose": object_pose,
        "allegro_hand_qpos": torch.tensor(allegro_hand_qpos),
        "shadow_hand_qpos": torch.tensor(shadow_hand_qpos),
        "schunk_svh_hand_qpos": torch.tensor(schunk_svh_hand_qpos),
        "leap_hand_qpos": torch.tensor(leap_hand_qpos),
        "ability_hand_qpos": torch.tensor(ability_hand_qpos),
        "panda_gripper_qpos": torch.tensor(panda_gripper_qpos),
        "gripper_qpos": torch.tensor(gripper_qpos),
        "background_with_obj_names": background_with_obj_names,
        "background_with_obj_ids": background_with_obj_ids,
        "grasped_with_obj_name": grasped_with_obj_name,
        "grasped_with_obj_id": grasped_with_obj_id,
        "grasped_obj_pose":grasped_obj_pose,
        "grasped_obj_idx":grasped_obj_idx,
        "grasped_obj_point3d":torch.tensor(grasped_obj_point3d, dtype=torch.float32), # nx3
    }
    return result


def load_dataset_single(file):
    # use provided file path
    data = np.load(file, allow_pickle=True)
    return data['data']


class HandDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item, np.void):
            item = item.item()
        x = torch.as_tensor(item["hand_pose"], dtype=torch.float32).reshape(-1)
        ydict = {}
        for k, v in item.items():
            if k.endswith('_qpos'):
                ydict[k] = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
        return x, ydict

