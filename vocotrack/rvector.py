import torch
import torch.nn as nn
from .resnet import ResNet
from .config import RVectorCfg
from speechbrain.nnet.pooling import StatisticsPooling


class RVector(nn.Module):
    def __init__(self, config: RVectorCfg):
        super().__init__() 
        config = RVectorCfg.model_validate(config)

        self.resnet = resnet = ResNet(config.res_net)
        self.pooling = StatisticsPooling()

        size = 2 * resnet.out_channels * 3
        
        self.emb_gender = nn.Embedding(2, config.emb_gender)
        self.emb_word = nn.Embedding(100, config.emb_word)
        hidden_dim = size + config.emb_gender + config.emb_word
        
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(hidden_dim, 256)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        mfcc: torch.Tensor,
        lengths: torch.Tensor,
        gender: torch.Tensor,
        word_id: torch.Tensor,
    ) -> torch.Tensor:
        x = self.resnet(mfcc)
        lengths = (lengths // self.resnet.total_stride).clamp_min(1)
        
        x = x.transpose(1, 2)
        x = x.transpose(1, 3)
    
        x = self.pooling(x, lengths)

        x = self.flatten(x)
        meta = torch.cat([self.emb_gender(gender), self.emb_word(word_id)], dim=1)

        x = torch.cat([x, meta], dim=1)
        x = self.fc(x)
        
        output = self.relu(x)
        
        return output
