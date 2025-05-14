import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
from vocotrack import RVectorCfg, RVectorClassifier
from dataset import Dataset
from typing import Any
from torchmetrics import AveragePrecision
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
        weight_decay: float = 1e-4,
        factor: float = 0.5,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        config = RVectorCfg.model_validate(config)
        config = config.model_dump(mode='json')
        
        self.model = RVectorClassifier(config)
        self.val_auc_pr = AveragePrecision(task='binary')
        
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, mfcc: torch.Tensor, length: torch.Tensor, gender: torch.Tensor, word_id: torch.Tensor) -> torch.Tensor:
        return self.model(mfcc, length, gender, word_id)
    
    def _predict(
        self, 
        mfcc: torch.Tensor, 
        length: torch.Tensor, 
        gender: torch.Tensor, 
        word_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(mfcc, length, gender, word_id)
        
        probs = torch.softmax(logits, dim=1)     
        
        return probs, logits
    
    def _step(self, batch: Dataset, batch_idx: int) -> tuple[float, torch.Tensor]:
        mfcc     = batch['mfcc']
        length  = batch['length']
        gender   = batch['gender']
        word_id  = batch['word_id']
        label   = batch['label']
        
        probs, logits = self._predict(mfcc, length, gender, word_id)
        
        loss  = self.criterion(logits, label)
        
        return loss, probs
    
    def training_step(self, batch: Dataset, batch_idx: int) -> float:
        loss, _ = self._step(batch, batch_idx)
        
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, probs = self._step(batch, batch_idx)
        self.log('val_loss', loss)
        
        pos_probs = probs[:, 1]
        self.val_auc_pr.update(pos_probs, batch['label'])
        return loss
    
    def on_validation_epoch_end(self) -> None:
        auc_pr = self.val_auc_pr.compute()
        self.log('val_auc_pr', auc_pr)
        self.val_auc_pr.reset()
    
    def test_step(self, batch: Dataset, batch_idx: int) -> float:
        loss, _ = self._step(batch, batch_idx)
        
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams['factor'],
            patience=2,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def predict_step(self, batch: Dataset, batch_idx: int) -> Prediction:
        mfcc    = batch['mfcc']
        length  = batch['length']
        gender  = batch['gender']
        word_id = batch['word_id']
        
        logits = self(mfcc, length, gender, word_id)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        
        return Prediction(preds=preds, probs=probs)
        