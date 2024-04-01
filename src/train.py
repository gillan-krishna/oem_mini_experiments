import lightning as pl
from data.dataset import OEMMiniDataLoader
from lightning.pytorch.loggers import NeptuneLogger
from networks.UNet import LitUNet
from torch import Tensor

if __name__ == '__main__':
    BATCH_SIZE = 8
    pl.seed_everything(42, workers=True)
    class_weights = Tensor([0.9829, 0.7839, 0.8128, 0.9326, 0.8154, 0.9614, 0.8843, 0.8267]).cuda()
    model = LitUNet(arch="unetplusplus", encoder_name="resnet18", attention=True, lr=3e-4, focal_weights=class_weights)
    oem_mini = OEMMiniDataLoader(batch_size=BATCH_SIZE, num_classes=9)
    neptune_logger = NeptuneLogger(
        project='gillan-k/mini-oem',
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYWRiZGVlNC04NjA2LTRlMmYtODE4OS0zYWQ4NjFhYTEyMDIifQ==",
        log_model_checkpoints=False,
    )

    trainer = pl.Trainer(
        logger=neptune_logger,
        accelerator="gpu",
        deterministic=True,
        max_epochs=500,
        # profiler="pytorch",
        accumulate_grad_batches=4,
        # enable_checkpointing=True,
        # fast_dev_run=True,
        # overfit_batches=1,
    )
    trainer.fit(model=model, datamodule=oem_mini, ckpt_path='/home/ubuntu/hrl/oem_mini_experiments/.neptune/Untitled/MIN-39/checkpoints/epoch=175-step=3696.ckpt')