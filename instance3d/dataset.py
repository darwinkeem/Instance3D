import os
import csv
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class Instance(Dataset):
    def __init__(self, root='./data', split='train', transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.csv_loc = ''
        self.items = []

        if self.split == 'train':
            self.csv_loc = os.path.join(self.root, 'train_PF.csv')
        elif self.split == 'test':
            self.csv_loc = os.path.join(self.root, 'test_PF.csv')

        f = open(self.csv_loc, 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for line in rdr:
            self.items.append(line)
        f.close()

    def __getitem__(self, index):
        X = self.items[index][1:-1]
        X = torch.from_numpy(X)
        X = X.view(4, 8, 8)

        y = self.items[index][-1]
        y = torch.from_numpy(y)

        sample = (X, y)

        return sample

    def __len__(self):
        return len(self.items)