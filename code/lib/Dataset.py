import numpy as np
from torch.utils.data import Dataset


class FMDataset(Dataset):
    def __init__(self, data):
        self.feature = data['feature']
        self.rating = data['rating']
        self.user = data['user']

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):

        return self.rating[idx], self.user[idx]
