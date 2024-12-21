import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from utils import *
from constants import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from datasets.lingo import LingoDataset

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))


@hydra.main(version_base=None, config_path="config", config_name="config_train_lingo")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus
    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, cfg):

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)

    synhsi_dataset = LingoDataset(**cfg.dataset)

    sampler = DistributedSampler(synhsi_dataset)
    dataloader = DataLoader(synhsi_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True)

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    trainer.set_dataset_and_model(synhsi_dataset, model)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))

    for epoch in range(cfg.epochs):
        print(f'Start epoch {epoch}', flush=True)
        sampler.set_epoch(epoch)
        step = 0
        for batch in dataloader:
            step += 1
            optimizer.zero_grad()

            joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco = batch
            joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco = joints.to(device), \
                                                                                                        mat.to(device), scene_flag.to(device), \
                                                                                                        text_clip_embedding.to(device), \
                                                                                                        pelvis_goal.to(device), hand_goal.to(device), \
                                                                                                        is_pick.to(device), need_scene.to(device), need_pelvis_dir.to(device), pi.to(device), \
                                                                                                        need_pi.to(device), is_loco.to(device)

            t = torch.randint(0, trainer.timesteps, (cfg.batch_size,), device=device).long()
            with torch.no_grad():
                mask, _, _ = get_mask(joints, -1, p=1., fixed_frame=cfg.auto_regre_num)

            loss = trainer.p_losses(joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco)

            if step % 10 == 0:
                print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}", flush=True)
                if cfg.use_tensorboard and rank == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)

            loss.backward()
            optimizer.step()

        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))

        torch.distributed.barrier()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()


def get_mask(x_start, ind, p, fixed_frame=0, mask_y=True):
    '''
    get mask for the input sequence of pre frames and final goal frame
    '''
    mask_frame = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)
    mask_goal = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)

    # goal mask
    if ind != -1:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_goal[rand_batch, -1, ind * 3: ind * 3 + 3] = True
        if not mask_y:
            mask_goal[rand_batch, -1, ind * 3 + 1] = False

    # prefix frame mask
    if fixed_frame > 0:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_frame[rand_batch, :fixed_frame, :] = True
    mask = torch.logical_or(mask_frame, mask_goal)
    return mask, mask_frame, mask_goal


if __name__ == '__main__':
    train()
