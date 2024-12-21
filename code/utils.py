import torch
import hydra
import numpy as np
from einops import rearrange
import random
import os
import smplx
from constants import SMPL_DIR, relaxed_hand_pose, pelvis_shift
from torch import nn
import transforms as T
import pickle as pkl


def joints_to_smpl_hand(model, joints, joints_ind, hand_ind, optim_norm, min_dist, hand_loss_mask):

    input_len = joints.shape[0]
    joints = joints.reshape(input_len, -1, 3)
    joints = joints.permute(1, 0, 2)
    trans_np = joints[0].detach().cpu().numpy()
    joints = joints - joints[0]
    joints = joints.permute(1, 0, 2)
    joints = joints.reshape(input_len, -1)
    pose_pred = model(joints)
    pose_pred = pose_pred.reshape(-1, 6)
    pose_pred = T.matrix_to_axis_angle(T.rotation_6d_to_matrix(pose_pred)).reshape(input_len, -1)

    pose_output, transl, left_hand, right_hand, joints_output = optimize_smpl_hand(pose_pred, joints, joints_ind, hand_ind, optim_norm, min_dist, hand_loss_mask)

    transl = trans_np - np.array(pelvis_shift) + transl
    joints_output += trans_np.reshape(-1, 1, 3) - np.array(pelvis_shift).reshape(1, 1, 3)
    return pose_output, transl, left_hand, right_hand, joints_output


def optimize_smpl_hand(pose_pred, joints, joints_ind, hand_ind, optim_norm, min_dist, hand_loss_mask, hand_pca=45):
    device = joints.device
    len = joints.shape[0]

    smpl_model = smplx.create(SMPL_DIR, model_type='smplx',
                              gender='male', ext='npz',
                              num_betas=10,
                              use_pca=False,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=len,
                              ).to(device)
    smpl_model.eval()

    joints = joints.reshape(len, -1, 3) + torch.tensor(pelvis_shift).to(device)
    pose_input = torch.nn.Parameter(pose_pred.detach(), requires_grad=True)
    transl = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], 3).to(device), requires_grad=True)
    left_hand = torch.from_numpy(relaxed_hand_pose[:45].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    right_hand = torch.from_numpy(relaxed_hand_pose[45:].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    optimizer = torch.optim.Adam(params=[pose_input, transl], lr=0.05)
    loss_fn = nn.MSELoss()


    for step in range(100):
        # fit the smplx param to the original joints
        smpl_output = smpl_model(transl=transl, body_pose=pose_input[:, 3:], global_orient=pose_input[:, :3], return_verts=True,
                                 left_hand_pose=left_hand,
                                 right_hand_pose=right_hand,
                                 )
        joints_output = smpl_output.joints[:, joints_ind].reshape(len, -1, 3)
        loss = loss_fn(joints[:, :], joints_output[:, :])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('step: {}, loss: {}'.format(step, loss.item()))

    # hand target
    target_joints = joints_output.detach().clone()
    target_joints -= 1. * optim_norm * min_dist.unsqueeze(-1) * hand_loss_mask.unsqueeze(-1)

    # fit the smplx param to original joints meanwhile hand joints are close to the object
    for step in range(100, 200):
        smpl_output = smpl_model(transl=transl, body_pose=pose_input[:, 3:], global_orient=pose_input[:, :3], return_verts=True,
                                 left_hand_pose=left_hand,
                                 right_hand_pose=right_hand,
                                 )
        joints_output = smpl_output.joints[:, joints_ind].reshape(len, -1, 3)


        loss = loss_fn(joints[:, :], joints_output[:, :]) + loss_fn(target_joints[:, :] * hand_loss_mask.unsqueeze(-1), joints_output[:, :] * hand_loss_mask.unsqueeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('step: {}, loss: {}'.format(step, loss.item()))



    return pose_input.detach().cpu().numpy(), transl.detach().cpu().numpy(), left_hand.detach().cpu().numpy(), right_hand.detach().cpu().numpy(), joints_output.detach().cpu().numpy()


def seed_everything(seed: int) -> object:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def transform_points(x, mat):
    shape = x.shape
    x = rearrange(x, 'b t (j c) -> b (t j) c', c=3)  # B x N x 3
    x = torch.einsum('bpc,bck->bpk', mat[:, :3, :3], x.permute(0, 2, 1))  # B x 3 x N   N x B x 3
    x = x.permute(2, 0, 1) + mat[:, :3, 3]
    x = x.permute(1, 0, 2)
    x = x.reshape(shape)

    return x


def create_meshgrid(bbox, size, batch_size=1):
    x = torch.linspace(bbox[0], bbox[1], size[0])
    y = torch.linspace(bbox[2], bbox[3], size[1])
    z = torch.linspace(bbox[4], bbox[5], size[2])
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    grid = grid.repeat(batch_size, 1, 1)

    return grid


def zup_to_yup(coord):
    # change the coordinate from z-up to y-up
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 2] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[2] *= -1

    return coord


def yup_to_zup(coord):
    # change the coordinate from y-up to z-up
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 1] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[1] *= -1

    return coord

def rigid_transform_3D(A, B, scale=False):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        # return None, None, None
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t


def find_free_port():
    from contextlib import closing
    import socket

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def init_model(model_cfg, device, eval, load_state_dict=False, need_ddp=True):
    model = hydra.utils.instantiate(model_cfg)
    if eval:
        load_state_dict_eval(model, model_cfg.ckpt, device=device)
    else:
        model = model.to(device)
        if need_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False,
                                                              find_unused_parameters=True)
        if load_state_dict:
            model.module.load_state_dict(torch.load(model_cfg.ckpt))
            model.train()

    return model


def load_state_dict_eval(model, state_dict_path, map_location='cuda:0', device='cuda'):
    state_dict = torch.load(state_dict_path, map_location=map_location)
    key_list = [key for key in state_dict.keys()]
    for old_key in key_list:
        new_key = old_key.replace('module.', '')
        state_dict[new_key] = state_dict.pop(old_key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()