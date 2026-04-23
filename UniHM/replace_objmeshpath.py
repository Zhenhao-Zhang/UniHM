import numpy as np
from glob import glob
from functools import partial

files = glob("/your/path/to/npzfiles/*.npz")
target_path_prefix = "/xxx/dexycb/models/"

def replace_path(raw_path, target_prefix):
     return target_prefix + "/".join(raw_path.split("/")[-2:])

replace_path_with_prefix = partial(replace_path, target_prefix=target_path_prefix)

for file in files:
    with np.load(file, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["object_mesh_file"] = np.asarray(
        [replace_path_with_prefix(item) for item in payload["object_mesh_file"]]
    )
    np.savez_compressed(file, **payload)
