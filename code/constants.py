import numpy as np
import os

ROOT_DIR = '..'
SMPL_DIR = os.path.join(ROOT_DIR, 'smpl_models')

OBJ_ACT_DICT = {
    'lie down': 0,
    'squat': 1,
    'mouse': 2,
    'keyboard': 3,
    'laptop': 4,
    'phone': 5,
    'book': 6,
    'bottle': 7,
    'pen': 8,
    'vase': 9,
}

SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]

rest_pelvis = np.matrix([[0.0000e+00,  0.0000e+00,  0.0000e+00],
                             [5.6144e-02, -9.4542e-02, -2.3475e-02],
                             [-5.7870e-02, -1.0517e-01, -1.6559e-02]])
pelvis_shift = [0.001144, -0.366919, 0.012666]

relaxed_hand_pose = np.array([0.11168, 0.04289, -0.41644,
                     0.10881, -0.06599, -0.75622,
                     -0.09639, -0.09092, -0.18846,
                     -0.1181, 0.05094, -0.52958,
                     -0.1437, 0.05524, -0.70486,
                     -0.01918, -0.09234, -0.33791,
                     -0.45703, -0.19628, -0.62546,
                     -0.21465, -0.066, -0.50689,
                     -0.36972, -0.06034, -0.07949,
                     -0.14187, -0.08585, -0.63553,
                     -0.30334, -0.05788, -0.63139,
                     -0.17612, -0.13209, -0.37335,
                     0.85096, 0.27692, -0.09155,
                     -0.49984, 0.02656, 0.05288,
                     0.53556, 0.04596, -0.27736,
                     0.11168, -0.04289, 0.41644,
                     0.10881, 0.06599, 0.75622,
                     -0.09639, 0.09092, 0.18846,
                     -0.1181, -0.05094, 0.52958,
                     -0.1437, -0.05524, 0.70486,
                     -0.01918, 0.09234, 0.33791,
                     -0.45703, 0.19628, 0.62546,
                     -0.21465, 0.066, 0.50689,
                     -0.36972, 0.06034, 0.07949,
                     -0.14187, 0.08585, 0.63553,
                     -0.30334, 0.05788, 0.63139,
                     -0.17612, 0.13209, 0.37335,
                     0.85096, -0.27692, 0.09155,
                     -0.49984, -0.02656, -0.05288,
                     0.53556, -0.04596, 0.27736]).astype(np.float32)