import torch
from src.config import *
from torch.utils.data import Dataset


class HedgePredDataset(Dataset):
    def __init__(self, pos_feats, neg_feats):
        # labeling
        x_pos = torch.from_numpy(pos_feats).float()
        y_pos = torch.ones(len(pos_feats), 1)

        x_neg = torch.from_numpy(neg_feats).float()
        y_neg = torch.zeros(len(neg_feats), 1)

        # dataset
        self.x_data = torch.cat([x_pos, x_neg], dim=0)
        self.y_data = torch.cat([y_pos, y_neg], dim=0)

        self.len = self.x_data.shape[0]

        # custom attributes
        n_pos = x_pos.shape[0]
        n_neg = x_neg.shape[0]

        p2n_ratio = n_pos / float(n_neg)
        weights_pos = torch.ones(n_pos)
        weights_neg = p2n_ratio * torch.ones(n_neg)

        self.sampling_weights = torch.cat([weights_pos, weights_neg], dim=0)
        self.n_pos = n_pos
        self.n_neg = n_neg

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
