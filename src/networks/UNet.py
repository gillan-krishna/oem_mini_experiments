import lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from custom_loss import TverskyFocalLoss, BalancedTverskyFocalLoss
import torch
from metrics import fscore, iou
from torch.nn.functional import softmax

torch.set_float32_matmul_precision("medium")

class LitUNet(pl.LightningModule):
    def __init__(self, arch='unetplusplus', encoder_name='resnet18', attention=True, lr=3e-3, focal_weights= None):
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
                activation="softmax2d",
                decoder_attention_type=attention_type,
            )
        else:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=9,
                activation="softmax2d",
                decoder_attention_type=attention_type,
            )
        
        if focal_weights is None:
            self.loss_fn = BalancedTverskyFocalLoss(ignore_index=0)
        else:
            self.loss_fn = TverskyFocalLoss(weight=focal_weights,ignore_index=0)
        
        # self.train_step_loss = []
        # self.train_fs = []
        # self.train_iou = []
        # self.val_step_loss = []
        # self.val_fs = []
        # self.val_iou = []
        self.lr = lr

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        fs = fscore(y_hat, y)
        iou_scores = iou(y_hat, y, num_classes=9)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, batch_size=8)
        for idx, f1 in enumerate(fs):
            self.log(f"train/f1_class_{idx}", f1, prog_bar=True, on_epoch=True, batch_size=8)
        for idx, iou_score in enumerate(iou_scores):
            self.log(f"train/iou_class_{idx}", iou_score, prog_bar=True, on_epoch=True, batch_size=8)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        fs = fscore(y_hat, y)
        iou_scores = iou(y_hat, y, num_classes=9)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=8)
        for idx, f1 in enumerate(fs):
            self.log(f"val/f1_class_{idx}", f1, prog_bar=True, on_epoch=True, batch_size=8)
        for idx, iou_score in enumerate(iou_scores):
            self.log(f"val/iou_class_{idx}", iou_score, prog_bar=True, on_epoch=True, batch_size=8)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        preds = torch.argmax(softmax(input=y_hat, dim=1), dim=1).permute(1,2,0).cpu().numpy()
        true = torch.argmax(y, dim=1).permute(1,2,0).cpu().numpy()
        logits = softmax(input=y_hat, dim=1).permute(2,3,1,0).cpu().numpy()
        return preds, true, logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer