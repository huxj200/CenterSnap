import json
import os
import numpy as np

def get_scale_factors(data_dir):
    path_to_modelsinfo = os.path.join(data_dir, 'obj_models\\models\\models_info.json')

    scale_factors = {}
    with open(path_to_modelsinfo, 'r') as f:
        objects_dict = json.load(f)
    for key in objects_dict.keys():
        info = objects_dict[key]
        size_x = info['size_x']
        size_y = info["size_y"]
        size_z = info["size_z"]
        bbox_dims = [size_x, size_y, size_z]
        scale_factors[key] = np.linalg.norm(bbox_dims)
    return scale_factors