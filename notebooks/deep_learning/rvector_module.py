import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
from vocotrack import RVectorCfg, RVector
from dataset import Dataset
from pydantic import BaseModel, ConfigDict


class Prediction(BaseModel):
    preds: torch.Tensor
    probs: torch.Tensor
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RVectorModule(LightningModule):
    def __init__(
        self, 
        config: RVectorCfg, 
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        
        config = RVectorCfg.model_validate(config)
        config = config.model_dump(mode='json')
        
        self.rvector = RVector(config)
        self.linear = nn.Linear(256, 100)
        
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, mfcc: torch.Tensor, lengths: torch.Tensor, gender: torch.Tensor, word_id: torch.Tensor) -> torch.Tensor:
        x = self.rvector(mfcc, lengths, gender, word_id)
        return self.linear(x)
    
    def _step(self, batch: Dataset, batch_idx: int) -> float:
        mfcc     = batch['mfcc']
        length  = batch['length']
        gender   = batch['gender']
        word_id  = batch['word_id']
        label   = batch['label']
        
        preds = self(mfcc, length, gender, word_id)
        
        loss  = self.criterion(preds, label)
        
        return loss
    
    def training_step(self, batch: Dataset, batch_idx: int) -> float:
        loss = self._step(batch, batch_idx)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Dataset, batch_idx: int) -> float:
        loss = self._step(batch, batch_idx)
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        return optimizer
    
    def predict_step(self, batch: Dataset, batch_idx: int) -> Prediction:
        mfcc    = batch['mfcc']
        length  = batch['length']
        gender  = batch['gender']
        word_id = batch['word_id']
        
        logits = self(mfcc, length, gender, word_id)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        
        return Prediction(preds=preds, probs=probs)
        