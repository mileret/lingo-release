import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import pickle as pkl


class SchedulerDataset(Dataset):
    def __init__(self, folder, device, mesh_grid, batch_size, step, nb_voxels, train=True,
                 load_scene=True, load_language=True, load_pelvis_goal=False, load_hand_goal=False,
                 max_window_size=16,
                 use_pi=True,
                 test_scene_name=None,
                 **kwargs):

        self.folder = folder
        self.device = device
        self.train = train
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_pelvis_goal = load_pelvis_goal
        self.load_hand_goal = load_hand_goal
        self.use_pi = use_pi
        self.test_scene_name = test_scene_name
        self.max_window_size = max_window_size

        self.global_orient = np.load(os.path.join(folder, 'human_orient.npy'))
        self.joints = np.load(os.path.join(folder, 'human_joints_aligned.npy'))

        if self.load_language:
            if self.max_window_size == 16:
                language_motion_dict_filename = 'scheduler_inter16.pkl'

            with open(os.path.join(self.folder, 'language_motion_dict', language_motion_dict_filename), 'rb') as f:
                language_motion_dict = pkl.load(f)
            self.end_range = language_motion_dict['end_range']
            self.text = language_motion_dict['text']

            self.clip_features = np.load(
                os.path.join(self.folder, 'clip_features.npy'))
            with open(os.path.join(self.folder, 'text2features_idx.pkl'), 'rb') as f:
                self.text2features_idx = pkl.load(f)

            self.need_scene = language_motion_dict['need_scene']
            self.need_pelvis_dir = language_motion_dict['need_pelvis_dir']
            self.pi = language_motion_dict['pi']
            self.need_pi = language_motion_dict['need_pi']
            self.left_hand_inter_frame = language_motion_dict['left_hand_inter_frame']
            self.right_hand_inter_frame = language_motion_dict['right_hand_inter_frame']

            self.start_ind = language_motion_dict['start_idx']
            self.end_ind = language_motion_dict['end_idx']

        self.step = step
        self.batch_size = batch_size

        if self.load_scene:
            self.mesh_grid = mesh_grid
            self.nb_voxels = nb_voxels
            self.scene_occ = []
            self.scene_dict = {}
            with open(os.path.join(folder, 'scene_name.pkl'), 'rb') as f:
                self.scene_name = pkl.load(f)
            if train:
                self.scene_folder = os.path.join(folder, 'Scene')
            else:
                self.scene_folder = os.path.join(folder, 'Scene_vis')
            scene_file_list = sorted(os.listdir(self.scene_folder))
            if not train:
                scene_file_list = [file for file in scene_file_list if file.split('.')[0] == self.test_scene_name]

            for sid, file in enumerate(scene_file_list):
                print(f"{sid} Loading Scene Mesh {file}")
                scene_occ = np.load(os.path.join(self.scene_folder, file))
                scene_occ = torch.from_numpy(scene_occ).to(device=device, dtype=bool)
                self.scene_dict[file[:-4]] = sid

            if train:
                self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
                self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
            else:
                self.scene_grid_np = np.array([-4, 0, -6, 4, 2, 6, 400, 100, 600])
                self.scene_grid_torch = torch.tensor([-4, 0, -6, 4, 2, 6, 400, 100, 600]).to(device)


            self.batch_id = torch.linspace(0, batch_size - 1, batch_size).tile((nb_voxels[0]*nb_voxels[1]*nb_voxels[2], 1)).T \
                .reshape(-1, 1).to(device=device, dtype=torch.long)

        if self.max_window_size == 16:
            norm = np.load(os.path.join(folder, 'norm_inter_and_loco__16frames.npy'))


        self.min = norm[0].astype(np.float32)
        self.max = norm[1].astype(np.float32)
        self.min_torch = torch.tensor(self.min).to(device)
        self.max_torch = torch.tensor(self.max).to(device)

    def __getitem__(self, idx):
        if self.load_language:
            start_idx = int(self.start_ind[idx])
            end_idx = int(self.end_ind[idx])
            assert end_idx - start_idx == self.max_window_size * 3

            pelvis_goal = np.zeros((3, )).astype(np.float32)
            hand_goal = np.zeros((3, )).astype(np.float32)
            is_loco = False

            text = self.text[idx][0]
            text_clip_embedding = self.clip_features[[self.text2features_idx[text]]]  # (1, 768)
            text_clip_embedding = torch.from_numpy(text_clip_embedding).float()  # (1, 768)
            text_clip_embedding = text_clip_embedding / torch.norm(text_clip_embedding, dim=1, keepdim=True)

            left_hand_inter_frame = self.left_hand_inter_frame[idx]
            right_hand_inter_frame = self.right_hand_inter_frame[idx]

            if left_hand_inter_frame != -1:
                hand_goal = self.joints[left_hand_inter_frame, 24].copy()  # left hand index1
            elif right_hand_inter_frame != -1:
                hand_goal = self.joints[right_hand_inter_frame, 26].copy()  # right hand index1

            need_pelvis_dir = self.need_pelvis_dir[idx]
            pi = self.pi[idx]
            need_pi = self.need_pi[idx]
            if need_pi:
                pi = pi + np.random.randint(-5, 5)
                if pi < 0:
                    pi = 0

            if need_pelvis_dir:
                if 'sit down' in text or 'lie down' in text:
                    pelvis_goal = self.joints[int(self.end_range[idx]), 0].copy()
                else:
                    pelvis_goal = self.joints[end_idx-3, 0].copy()
                    is_loco = True
                pelvis_goal[1] = 0.

        joints = self.joints[start_idx: end_idx: self.step]
        init_joints = np.array([joints[0, 0, 0], 0., joints[0, 0, 2]])
        joints = joints - init_joints
        pelvis_goal = pelvis_goal - init_joints
        hand_goal = hand_goal - init_joints

        global_orient = self.global_orient[start_idx: end_idx: self.step]
        init_global_orient = global_orient[0]
        init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
        shift_euler = np.array([0, 0, -init_global_orient_euler[2]])
        shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix()

        mat = np.eye(4)
        mat[:3, :3] = np.linalg.inv(shift_rot_matrix.T).T
        mat[:3, 3] = init_joints
        mat = mat.astype(np.float32)

        joints = joints @ shift_rot_matrix.T
        pelvis_goal = pelvis_goal @ shift_rot_matrix.T
        hand_goal = hand_goal @ shift_rot_matrix.T

        if is_loco:
            pelvis_goal_norm = np.linalg.norm(pelvis_goal)
            if pelvis_goal_norm >= 0.8:
                pelvis_goal = pelvis_goal / pelvis_goal_norm * 0.8

        joints = self.normalize(joints)
        joints = joints.astype(np.float32).reshape((joints.shape[0], -1))

        stop = self.end_range[idx] <= end_idx

        if not self.use_pi:
            pi = 0
            need_pi = False

        return joints.astype(np.float32), float(stop), int(pi), text_clip_embedding


    def __len__(self):
        return len(self.start_ind)

    def normalize(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = -1. + 2. * (data - self.min) / (self.max - self.min)
        data = data.reshape(shape_orig)

        return data

    def normalize_torch(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = -1. + 2. * (data - self.min_torch) / (self.max_torch - self.min_torch)
        data = data.reshape(shape_orig)

        return data

    def denormalize(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = (data + 1.) * (self.max - self.min) / 2. + self.min
        data = data.reshape(shape_orig)

        return data

    def denormalize_torch(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = (data + 1.) * (self.max_torch - self.min_torch) / 2. + self.min_torch
        data = data.reshape(shape_orig)

        return data

