import argparse

from instance3d.dataset import Instance
from instance3d.models import FC
from instance3d.trainer import Trainer

import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2
)
parser.add_argument(
    '--epoch', type=int, default=40
)
parser.add_argument(
    '--lr', type=float, default=0.001
)
parser.add_argument(
    '--dataset', type=str, default='./data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)
parser.add_argument(
    '--save_model', type=str, default='./save_model/'
)

cfg = parser.parse_args()
print(cfg)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    ds_train = Instance(root=cfg.dataset, split='train', transform=None)
    ds_test = Instance(root=cfg.dataset, split='test', transform=None)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    print("DATA LOADED")

    model = FC()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)