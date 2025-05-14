import torch
import torch.nn as nn
from .config import RVectorCfg
from .rvector import RVector

class RVectorClassifier(nn.Module):
    def __init__(self, config: RVectorCfg, num_classes: int = 2):
        super().__init__()
        self.rvector = RVector(config)
        self.linear = nn.Linear(256, num_classes)
        
    def forward(self, mfcc: torch.Tensor, lengths: torch.Tensor, gender: torch.Tensor, word_id: torch.Tensor) -> torch.Tensor:
        x = self.rvector(mfcc, lengths, gender, word_id)
        return self.linear(x)