import lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassF1Score
from custom_loss import TverskyFocalLoss, BalancedTverskyFocalLoss
import torch
from metrics import fscore
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
        
        # self.f1 = MulticlassF1Score(num_classes=9, average=None).to('cpu')
        self.train_step_loss = []
        self.train_fs = []
        # self.train_epoch_preds = []
        # self.train_epoch_targets = []
        self.val_step_loss = []
        self.val_fs = []
        # self.val_epoch_preds = []
        # self.val_epoch_targets = []
        self.lr = lr

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        fs = fscore(y_hat, y)
        self.train_step_loss.append(loss.detach().cpu())
        self.train_fs.append(fs.detach().cpu())
        # self.train_epoch_preds.append(y_hat.detach().cpu())
        # self.train_epoch_targets.append(y.detach().cpu())
        return loss
    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.train_step_loss).mean().cpu()
        self.train_step_loss.clear()
        self.log("train/mean_loss", epoch_mean, prog_bar=True, on_epoch=True)
        # epoch_preds = torch.cat(self.train_epoch_preds, dim=0).cpu()
        # self.train_epoch_preds.clear()
        # epoch_targets = torch.cat(self.train_epoch_targets, dim=0).cpu()
        # self.train_epoch_targets.clear()
        # f1_score = self.f1(epoch_preds, epoch_targets).cpu()
        # del epoch_preds
        # del epoch_targets
        
        # self.log_dict({'train/f1_Bareland':f1_score[1], 
        #                'train/f1_Rangeland':f1_score[2],
        #                'train/f1_Developed':f1_score[3],
        #                'train/f1_Road':f1_score[4],
        #                'train/f1_Tree':f1_score[5],
        #                'train/f1_Water':f1_score[6],
        #                'train/f1_Agri':f1_score[7],
        #                'train/f1_Building':f1_score[8]}, prog_bar=False, on_epoch=True)

        f1score = torch.stack(self.train_fs).mean().cpu()
        self.train_fs.clear()
        self.log('train/fscore', f1score, prog_bar=True, on_epoch=True)

    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.val_step_loss.append(loss.detach().cpu())
        # self.val_epoch_preds.append(y_hat.detach().cpu())
        # self.val_epoch_targets.append(y.detach().cpu())
        fs = fscore(y_hat, y)
        self.val_fs.append(fs.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        epoch_mean = torch.stack(self.val_step_loss).mean().cpu()
        self.val_step_loss.clear()
        self.log("val/mean_loss", epoch_mean, prog_bar=True, on_epoch=True)
        # epoch_preds = torch.cat(self.val_epoch_preds, dim=0).cpu()
        # self.val_epoch_preds.clear()
        # epoch_targets = torch.cat(self.val_epoch_targets, dim=0).cpu()
        # self.val_epoch_targets.clear()
        # f1_score = self.f1(epoch_preds, epoch_targets).cpu()
        # del epoch_preds
        # del epoch_targets
        
        # self.log_dict({'val/f1_Bareland':f1_score[1], 
        #                'val/f1_Rangeland':f1_score[2],
        #                'val/f1_Developed':f1_score[3],
        #                'val/f1_Road':f1_score[4],
        #                'val/f1_Tree':f1_score[5],
        #                'val/f1_Water':f1_score[6],
        #                'val/f1_Agri':f1_score[7],
        #                'val/f1_Building':f1_score[8]}, prog_bar=False, on_epoch=True)
        f1score = torch.stack(self.val_fs).mean().cpu()
        self.val_fs.clear()
        self.log('val/fscore', f1score, prog_bar=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        preds = torch.argmax(softmax(input=y_hat, dim=1), dim=1).permute(1,2,0).cpu().numpy()
        true = torch.argmax(y, dim=1).permute(1,2,0).cpu().numpy()
        # print(preds.shape)
        # print(softmax(input=y_hat, dim=1).shape
        logits = softmax(input=y_hat, dim=1).permute(2,3,1,0).cpu().numpy()
        return preds, true, logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer