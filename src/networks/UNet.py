import lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassF1Score
from custom_loss import TverskyFocalLoss, BalancedTverskyFocalLoss
import torch

class LitUNet(pl.LightningModule):
    def __init__(self, arch='unetplusplus', encoder_name='resnet18', attention=True, lr=3e-4, focal_weights= None):
        super().__init__()
        self.save_hyperparameters()

        if attention:
            attention_type = 'scse'
        else:
            attention = None

        if arch == 'unet':
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=9,
                activation="softmax",
                decoder_attention_type=attention_type,
            )
        else:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=9,
                activation="softmax",
                decoder_attention_type=attention_type,
            )
        
        if focal_weights is None:
            self.loss_fn = BalancedTverskyFocalLoss(ignore_index=0)
        else:
            self.loss_fn = TverskyFocalLoss(weight=focal_weights,ignore_index=0)
        
        self.f1 = MulticlassF1Score(num_classes=9, average=None)
        self.train_step_loss = []
        self.train_epoch_preds = []
        self.train_epoch_targets = []
        self.val_step_loss = []
        self.val_epoch_preds = []
        self.val_epoch_targets = []
        self.lr = lr

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.train_step_loss.append(loss.detach())
        self.train_epoch_preds.append(y_hat.cpu().numpy())
        self.train_epoch_targets.append(y.cpu().numpy())
        return loss
    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.train_step_loss).mean()
        epoch_preds = torch.cat(self.epoch_preds, dim=0)
        epoch_targets = torch.cat(self.epoch_targets, dim=0)
        f1_score = self.f1(epoch_preds, epoch_targets)
        self.log("train/mean_loss", epoch_mean, prog_bar=True, on_epoch=True)
        self.log('train/f1')
        self.train_step_loss.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.val_step_loss.append(loss)
        return loss

    def on_validation_epoch_end(self):
        epoch_mean = torch.stack(self.val_step_loss).mean()
        self.log("val/mean_loss", epoch_mean, prog_bar=True, on_epoch=True)
        self.val_step_loss.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer