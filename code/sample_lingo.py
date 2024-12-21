import os
import pickle as pkl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from models.synhsi import TimingModel
from models.joints_to_smplx import joints_to_smpl
from utils import *
from constants import *
from clip_utils import get_clip_features
from datasets.lingo import LingoDataset
from astar import get_path

def sample_step(cfg, step, mat, fixed_points, sampler, cond, trajectory, pi):
    raw_text = cond['raw_text']
    text_emb = cond['text_emb']
    pelvis_goal = cond['pelvis_goal']
    pelvis_goal = transform_points(pelvis_goal.reshape(1, 1, 3), torch.inverse(mat)) # convert to local coordinates
    hand_goal = cond['hand_goal']
    hand_goal = transform_points(hand_goal.reshape(1, 1, 3), torch.inverse(mat))
    is_pick = cond['is_pick']

    need_scene = cond['need_scene']
    need_pelvis_dir = cond['need_pelvis_dir']
    is_loco = cond['is_loco']
    need_pi = cond['need_pi']

    speed_new = None
    if is_loco:
        pi = torch.zeros((cfg.batch_size, ), dtype=torch.long).to(cfg.device)

        curr_loc = mat[0, :3, 3].cpu().numpy()
        curr_loc = np.array([curr_loc[0], curr_loc[2]]).reshape(1, 2)

        dist = np.linalg.norm(curr_loc - trajectory, axis=1)
        min_idx = np.argmin(dist)
        pelvis_goal = torch.tensor([trajectory[min(min_idx+50, len(trajectory)-1)][0], 0,
                                    trajectory[min(min_idx+50, len(trajectory)-1)][1]]).reshape(1, 1, 3).to(cfg.device).float()

        pelvis_goal = transform_points(pelvis_goal, torch.inverse(mat)) # convert to local coordinates

        pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
        if pelvis_goal_norm >= cfg.speed:
            pelvis_goal = pelvis_goal / pelvis_goal_norm * cfg.speed

        theta_list = [[np.pi/18*i, -np.pi/18*i] for i in range(18)]
        theta_list = np.array(theta_list).reshape(-1)

        for theta in theta_list:
            goal_points = pelvis_goal.reshape(1, 3).repeat(9, 1) # 9x3
            goal_points[:, 1] += torch.arange(0.1, 1, 0.1).to(cfg.device).reshape(-1) # 9x3
            goal_points = goal_points.squeeze(0).repeat(20, 1, 1)
            goal_points[:, :, [0, 2]] *= torch.arange(0, 1, 0.05).to(cfg.device).reshape(-1, 1, 1)

            goal_points = transform_points(goal_points, mat) # convert to global coordinates
            goal_occ = sampler.dataset.get_occ_for_points(goal_points, [0])

            if goal_occ.sum() < 5:
                # no obstacle in the way
                break

            rotation_matrix = R.from_euler('y', theta).as_matrix()
            rotation_matrix = torch.from_numpy(rotation_matrix).to(cfg.device).float()

            pelvis_goal = pelvis_goal.reshape(1, 3) @ rotation_matrix
            pelvis_goal = pelvis_goal.reshape(1, 1, 3)

        pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
        speed_factor = pelvis_goal[0, 0, 2] / pelvis_goal_norm
        speed_factor = (speed_factor + 1.0) / 4 + 0.5
        speed_new = speed_factor

        pelvis_goal = pelvis_goal * speed_new

    if not cfg.use_pi:
        need_pi = torch.zeros((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
        pi = torch.zeros((cfg.batch_size, ), dtype=torch.long).to(cfg.device)

    print(f'pelvis_goal: {pelvis_goal}', 'pi: ', pi, 'need_pi: ', need_pi, 'need_scene: ', need_scene, 'need_pelvis_dir: ', need_pelvis_dir, 'raw_text: ', raw_text, 'speed: ', speed_new)

    scene_flag = sampler.dataset.scene_dict[cond['scene_name']]
    scene_flag = torch.tensor([scene_flag]*cfg.batch_size).to(cfg.device)

    samples, occs = sampler.p_sample_loop(fixed_points, mat, scene_flag, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco)

    points_gene = samples[-1]
    points_orig = transform_points(sampler.dataset.denormalize_torch(points_gene), mat)

    info_dict = {
        'points_orig': points_orig.reshape(cfg.batch_size, cfg.max_window_size, 3*cfg.dataset.nb_joints),
        'pelvis_goal': transform_points(pelvis_goal, mat).reshape(cfg.batch_size, 3), # global coordinates
        'pi': pi,
        'need_pi': need_pi,
        'need_scene': need_scene,
        'need_pelvis_dir': need_pelvis_dir,
        'raw_text': raw_text,
        'speed': speed_new,
        'scene_flag': scene_flag,
        'hand_goal': hand_goal,
        'is_pick': is_pick,
        'occ': occs[-1],
    }

    return info_dict


def get_mat(cfg, points):
    pelvis_new = points[:, -cfg.auto_regre_num, :9].cpu().numpy().reshape(cfg.batch_size, 3, 3)
    trans_mats = np.repeat(np.eye(4)[np.newaxis, :, :], cfg.batch_size, axis=0)
    for ip, pn in enumerate(pelvis_new):
        _, ret_R, ret_t = rigid_transform_3D(np.matrix(pn), rest_pelvis, False)
        ret_t[1] = 0.0
        rot_euler = R.from_matrix(ret_R).as_euler('zxy')
        shift_euler = np.array([0, 0, rot_euler[2]])
        shift_rot_matrix2 = R.from_euler('zxy', shift_euler).as_matrix()
        trans_mats[ip, :3, :3] = shift_rot_matrix2
        trans_mats[ip, :3, 3] = ret_t.reshape(-1)
    mat = torch.from_numpy(trans_mats).to(device=cfg.device, dtype=torch.float32)

    return mat


def get_guidance(cfg, seg_id):
    cond = {}

    with open(cfg.input_path, 'rb') as f:
        input = pkl.load(f)
    input = input[seg_id]

    cond['scene_name'] = input['scene_name']
    cond['raw_text'] = input['text']
    cond['text_emb'] = get_clip_features(input['text']).reshape(cfg.batch_size, 1, -1).to(cfg.device)
    cond['pelvis_goal'] = input['end_location']
    cond['pelvis_goal'] = np.array([cond['pelvis_goal'][0], 0, cond['pelvis_goal'][2]]) # set y to 0
    cond['pelvis_goal'] = torch.from_numpy(cond['pelvis_goal'].astype(np.float32)).to(cfg.device)
    cond['hand_goal'] = torch.from_numpy(input['hand_location'].astype(np.float32)).to(cfg.device).reshape(cfg.batch_size, 1, 3)

    if 'pick up' in input['text'] or 'put down' in input['text']:
        cond['is_pick'] = torch.ones((cfg.batch_size, 1), dtype=torch.bool).to(cfg.device)
    else:
        cond['is_pick'] = torch.zeros((cfg.batch_size, 1), dtype=torch.bool).to(cfg.device)

    if 'walk' in cond['raw_text'] or 'sit down' in cond['raw_text'] or 'lie down' in cond['raw_text'] or 'type on ' in cond['raw_text'] or 'write on' in cond['raw_text'] or 'wash ' in cond['raw_text'] or 'punch ' in cond['raw_text'] or 'kick ' in cond['raw_text']:
        need_scene = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
    else:
        need_scene = torch.zeros((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)

    if 'walk' in cond['raw_text'] or 'sit down ' in cond['raw_text'] or 'lie down ' in cond['raw_text']:
        need_pelvis_dir = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
    else:
        need_pelvis_dir = torch.zeros((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)

    cond['need_scene'] = need_scene
    cond['need_pelvis_dir'] = need_pelvis_dir
    cond['is_loco'] = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device) if 'walk' in cond['raw_text'] else torch.zeros((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
    cond['need_pi'] = torch.zeros((cfg.batch_size, ), dtype=torch.bool).to(cfg.device) if 'walk' in cond['raw_text'] else torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)

    cond['start_location'] = input['start_location']
    cond['start_location'] = np.array([cond['start_location'][0], 0, cond['start_location'][2]])
    cond['start_location'] = torch.from_numpy(cond['start_location']).to(cfg.device)
    cond['episode_num'] = input['episode_num']
    cond['seg_num'] = input['seg_num']

    return cond


@hydra.main(version_base=None, config_path="config", config_name="config_sample_lingo")
def sample(cfg: DictConfig) -> None:
    device = cfg.device
    model_joints_to_smplx = init_model(cfg.model.model_smplx, device=device, eval=True)
    print('model_joints_to_smplx device: ', next(model_joints_to_smplx.parameters()).device)
    model_body = init_model(cfg.model.synhsi_body, device=device, eval=True)

    cond = get_guidance(cfg, 0)
    seg_num = cond['seg_num']
    cfg.dataset.test_scene_name = cond['scene_name']
    print(OmegaConf.to_yaml(cfg))

    synhsi_dataset = LingoDataset(**cfg.dataset)

    sampler_body = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler_body.set_dataset_and_model(synhsi_dataset, model_body)

    # load scheduler model
    if cfg.use_scheduler:
        scheduler_model = TimingModel(**cfg.model.scheduler)
        scheduler_model.load_state_dict(torch.load(cfg.scheduler_model_path, map_location=device))
        scheduler_model.to(device)
        scheduler_model.eval()
    else:
        scheduler_model = None

    points_all = []
    pi_list = []
    raw_text_list = []

    for seg_id in range(seg_num):
        if seg_id >= 1:
            cond = get_guidance(cfg, seg_id)
        seg_len = cond['episode_num']

        if seg_id == 0:
            stand_start_idx_list = [100]
            joints, mat, _, _, _, _, _, _, _, _, _, _ = sampler_body.dataset.__getitem__(stand_start_idx_list[0])
            joints = torch.from_numpy(joints).float().reshape(1, -1, cfg.dataset.nb_joints*3)
            mat = torch.from_numpy(mat).float().reshape(1, 4, 4)

            points, mat = joints.to(device), mat.to(device)
            points_orig = sampler_body.dataset.denormalize_torch(points) # (batch_size, max_window_size, nb_joints*3)

            theta = np.arctan2(-cond['pelvis_goal'].cpu().numpy()[2]+cond['start_location'].cpu().numpy()[2],
                                cond['pelvis_goal'].cpu().numpy()[0]-cond['start_location'].cpu().numpy()[0],) + np.pi/2
            rot_matrix = R.from_euler('y', theta).as_matrix()

            assert cfg.batch_size == 1
            mat[0, :3, :3] = torch.from_numpy(rot_matrix).to(device).float()
            points_orig = points_orig.reshape(cfg.batch_size, cfg.max_window_size, cfg.dataset.nb_joints, 3) @ mat[0, :3, :3].t()
            points_orig = points_orig.reshape(cfg.batch_size, cfg.max_window_size, cfg.dataset.nb_joints*3)

            translation_shift = points_orig[:, [-cfg.auto_regre_num], :3] - cond['start_location']
            translation_shift[0, 0, 1] = 0.
            points_orig = points_orig.reshape(cfg.batch_size, -1, cfg.dataset.nb_joints, 3)
            points_orig[:, :, :] -= translation_shift
            points_orig = points_orig.reshape(cfg.batch_size, -1, 3*cfg.dataset.nb_joints)
        else:
            points_orig = torch.from_numpy(points_all[-1].reshape(cfg.batch_size, -1, cfg.dataset.nb_joints*3)).to(cfg.device)

        if cond['is_loco']:
            if seg_id == 0:
                start_loc = cond['start_location'].cpu().numpy()[[0, 2]]
            else:
                start_loc = points_orig[:, -cfg.auto_regre_num].reshape(cfg.batch_size, cfg.dataset.nb_joints, 3)[0, 0, [0, 2]].cpu().numpy()
            end_loc = cond['pelvis_goal'].cpu().numpy()[[0, 2]]
            trajectory = get_path(start_loc, end_loc, sampler_body.dataset)
        else:
            trajectory = None

        # sample loop
        for step in tqdm(range(seg_len)):
            if step == 0:
                mat = get_mat(cfg, points_orig)
                fixed_points = points_orig[:, -cfg.auto_regre_num:].reshape(cfg.batch_size, cfg.auto_regre_num, cfg.dataset.nb_joints*3)
                fixed_points = sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))
            else:
                mat = get_mat(cfg, points)
                fixed_points = points[:, -cfg.auto_regre_num:].reshape(cfg.batch_size, cfg.auto_regre_num, cfg.dataset.nb_joints*3)
                fixed_points = sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

            phase = 1
            speed_inter = 3
            pi = torch.tensor([int((step + phase) * (cfg.max_window_size - cfg.auto_regre_num) * speed_inter)]).to(device=cfg.device, dtype=torch.long)
            pi_list.append(pi.cpu().numpy())
            raw_text_list.append(cond['raw_text'])

            info_dict = sample_step(cfg, step, mat, fixed_points, sampler_body, cond, trajectory, pi)
            points = info_dict['points_orig']  # points in global coordinates and is denormalized

            if step == seg_len - 1:
                if step == 0 and seg_id > 0:
                    points_all.append(points.cpu().numpy()[:, cfg.auto_regre_num:])
                else:
                    points_all.append(points.cpu().numpy())
            else:
                if step == 0 and seg_id > 0:
                    points_all.append(points.cpu().numpy()[:, cfg.auto_regre_num:-cfg.auto_regre_num])
                else:
                    points_all.append(points.cpu().numpy()[:, :-cfg.auto_regre_num])

            # scheduler
            if cfg.use_scheduler and not cond['is_loco']:
                points_loco = sampler_body.dataset.normalize_torch(transform_points(points, torch.inverse(mat)))
                stop_pred = scheduler_model(points_loco, cond['text_emb'], pi).squeeze(1)
                stop_pred = torch.sigmoid(stop_pred)
                if stop_pred > cfg.scheduler_threshold:                
                    break

            if cond['is_loco'] and seg_id != seg_num - 1:
                curr_loc = points[0, -1, :3].cpu().numpy().copy()
                curr_loc[1] = 0.0
                end_point = cond['pelvis_goal'].cpu().numpy().copy()
                dist2end = np.linalg.norm(curr_loc - end_point) # distance to the end point
                if dist2end < cfg.locomotion_threshold:
                    break


    points_all = np.concatenate(points_all, axis=1).reshape(cfg.batch_size, -1, cfg.dataset.nb_joints, 3)

    # save generated results
    exp_dir = cfg.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    for i in range(cfg.batch_size):
        keypoint_gene_torch = torch.from_numpy(points_all[i]).reshape(-1, cfg.dataset.nb_joints * 3).to(device)
        pose, transl, _, _ = joints_to_smpl(model_joints_to_smplx, keypoint_gene_torch, cfg.dataset.joints_ind, cfg.interp_s)
        output_data = {'transl': transl, 'body_pose': pose[:, 3:], 'global_orient': pose[:, :3],
                        'scene_name': cond['scene_name'], 'input_pkl_path': cfg.input_path,
                        'raw_text': raw_text_list,
                        }
        save_filename = f"output__{cfg.test_setting}__{cfg.repeat_time}.pkl"
        with open(os.path.join(exp_dir, save_filename), 'wb') as f:
            pkl.dump(output_data, f)
        print(f"Saved to {os.path.join(exp_dir, save_filename)}")

    print(cfg.test_setting, cfg.repeat_time)


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['ROOT_DIR'] = '../'

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    sample()
