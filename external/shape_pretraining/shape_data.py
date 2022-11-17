import os
import sys
import h5py
import glob
import numpy as np
import _pickle as cPickle
import argparse
from utils.shape_utils import sample_points_from_mesh

def save_model_to_hdf5(obj_model_dir, n_points, fps=False, include_distractors=False, with_normal=False):
    """ Save object models (point cloud) to HDF5 file.
        Dataset used to train the auto-encoder.
        Only use models from ShapeNetCore.
        Background objects are not inlcuded as default. We did not observe that it helps
        to train the auto-encoder.
    """
    # catId_to_synsetId = {1: '02876657', 2: '02880940', 3: '02942699', 4: '02946921', 5: '03642806', 6: '03797390'}
    # distractors_synsetId = ['00000000', '02954340', '02992529', '03211117']
    # with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'rb') as f:
    #     mug_meta = cPickle.load(f)
    # # read all the paths to models
    # print('Sampling points from mesh model ...')
    # print('dsd')
    if with_normal:
        train_data = np.zeros((3000, n_points, 6), dtype=np.float32)
        val_data = np.zeros((500, n_points, 6), dtype=np.float32)
    else:
        train_data = np.zeros((3000, n_points, 3), dtype=np.float32)
        val_data = np.zeros((500, n_points, 3), dtype=np.float32)
    train_label = []
    # val_label = []
    train_count = 0
    # val_count = 0

    path_to_mesh_models = glob.glob(os.path.join(obj_model_dir, 'models', '*.obj'))
    for mesh_model in sorted(path_to_mesh_models):
        model_points = sample_points_from_mesh(mesh_model, n_points, with_normal, fps=fps, ratio=2)
        model_points = model_points * np.array([[1.0, 1.0, -1.0]])
        train_data[train_count] = model_points
        train_label.append(1) #都为手机模型，此处类别就一直为 1
        train_count += 1

    num_train_instances = len(train_label)
    assert num_train_instances == train_count
    train_data = train_data[:num_train_instances]
    train_label = np.array(train_label, dtype=np.uint8)
    print('{} shapes found in train dataset'.format(num_train_instances))

    # write to HDF5 file
    print('Writing data to HDF5 file ...')
    if with_normal:
        filename = 'ShapeNetCore_{}_with_normal.h5'.format(n_points)
    else:
        filename = 'ShapeNetCore_{}.h5'.format(n_points)
    hfile = h5py.File(os.path.join(obj_model_dir, filename), 'w')
    train_dataset = hfile.create_group('train')
    train_dataset.attrs.create('len', num_train_instances)
    train_dataset.create_dataset('data', data=train_data, compression='gzip', dtype='float32')
    train_dataset.create_dataset('label', data=train_label, compression='gzip', dtype='uint8')
    hfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model_dir', type=str, required=True)
    args = parser.parse_args()
    obj_model_dir = args.obj_model_dir
    save_model_to_hdf5(obj_model_dir, n_points=2048, fps=True)