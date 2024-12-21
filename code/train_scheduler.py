import pdb

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import *
from constants import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from datasets.scheduler import SchedulerDataset
from models.synhsi import TimingModel

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))


@hydra.main(version_base=None, config_path="config", config_name="config_train_scheduler")
def train(cfg):
    rank = 0
    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)

    model = TimingModel(**cfg.model)
    model.to(device)
    model.train()

    scheduler_dataset = SchedulerDataset(**cfg.dataset)
    dataloader = DataLoader(scheduler_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            shuffle=True, pin_memory=True)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):

        print(f'Start epoch {epoch}', flush=True)
        step = 0
        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))

        for batch in dataloader:
            step += 1

            optimizer.zero_grad()

            joints, stop, pi, text_clip_embedding = batch
            joints, stop, pi, text_clip_embedding = joints.to(device), stop.to(device=device, dtype=torch.float32), pi.to(device), text_clip_embedding.to(device)

            stop_pred = model(joints, text_clip_embedding, pi).squeeze(1)

            loss = loss_fn(stop_pred, stop)

            if step % 10 == 1:
                print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}", flush=True)
                if cfg.use_tensorboard and rank == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)

                with torch.no_grad():
                    thres = 0.5
                    pred = stop_pred > thres
                    acc_0 = (pred[stop == 0] == stop[stop == 0]).float().mean().item()
                    acc_1 = (pred[torch.logical_not(stop==0)] == stop[torch.logical_not(stop==0)]).float().mean().item()
                    print(f"Accuracy on class 0: {acc_0}, class 1: {acc_1}", flush=True)

            loss.backward()
            optimizer.step()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()