import torch
from typing import Iterable
from torch.utils.data import Dataset

class Dataset(Dataset):
    __slots__ = ('mfcc', 'length', 'gender', 'word_id', 'label')
    
    def __init__(self, data: dict[str, torch.Tensor]):
        self.mfcc     = data['mfcc']
        self.length  = data['length']
        self.gender   = data['gender']
        self.word_id  = data['word_id']
        self.label   = data['label']

    def __len__(self):
        return self.label.size(0)

    def __getitem__(self, idx: Iterable[int]) -> dict[str, torch.Tensor]:
        return {
            'mfcc':     self.mfcc[idx],
            'length':  self.length[idx],
            'gender':   self.gender[idx].long(),
            'word_id':  self.word_id[idx].long(),
            'label':    self.label[idx],
        }
