from transformers import SegformerForSemanticSegmentation
from config import TrainingConfig, DatasetConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score
import torch.optim as optim
import pytorch_lightning as pl
from utils import MedicalSegmentationDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import ipdb 

def get_model(*, model_name, num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def dice_coef_loss(predictions, ground_truths, num_classes=2, dims=(1, 2), smooth=1e-8):
    """Smooth Dice coefficient + Cross-entropy loss function."""

    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)
    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)

    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (summation + smooth)
    dice_mean = dice.mean()

    CE = F.cross_entropy(predictions, ground_truths)

    return (1.0 - dice_mean) + CE

class MedicalSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        scheduler_name: str = "multistep_lr",
        num_epochs: int = 100,
    ):
        super().__init__()

        # Save the arguments as hyperparameters.
        self.save_hyperparameters()

        # Loading model using the function defined above.
        self.model = get_model(model_name=self.hparams.model_name, num_classes=self.hparams.num_classes)

        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")

    def forward(self, data):
        outputs = self.model(pixel_values=data, return_dict=True)
        upsampled_logits = F.interpolate(outputs["logits"], size=data.shape[-2:], mode="bilinear", align_corners=False)
        return upsampled_logits
    
    def training_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)

        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)
        
        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_f1(logits.detach(), target)

        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=False)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True, logger=False)
        return loss

    def on_train_epoch_end(self):
        # Computing and logging the training mean loss & mean f1.
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1", self.mean_train_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)

    def validation_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)
        
        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)

        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1.update(logits, target)

    def on_validation_epoch_end(self):
        
        # Computing and logging the validation mean loss & mean f1.
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1", self.mean_valid_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )

        LR = self.hparams.init_lr
        WD = self.hparams.weight_decay

        if self.hparams.optimizer_name in ("AdamW", "Adam"):
            optimizer = getattr(torch.optim, self.hparams.optimizer_name)(model.parameters(), lr=LR, 
                                                                          weight_decay=WD, amsgrad=True)
        else:
            optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WD)

        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.trainer.max_epochs // 2,], gamma=0.1)

            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "name": "multi_step_lr"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        else:
            return optimizer
        
if __name__=="__main__":
    pl.seed_everything(42, workers=True)

    # create model
    model = MedicalSegmentationModel(
        model_name=TrainingConfig.MODEL_NAME,
        num_classes=DatasetConfig.NUM_CLASSES,
        init_lr=TrainingConfig.INIT_LR,
        optimizer_name=TrainingConfig.OPTIMIZER_NAME,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        use_scheduler=TrainingConfig.USE_SCHEDULER,
        scheduler_name=TrainingConfig.SCHEDULER,
        num_epochs=TrainingConfig.NUM_EPOCHS,
    )

    # create data module
    data_module = MedicalSegmentationDataModule(
        num_classes=DatasetConfig.NUM_CLASSES,
        img_size=DatasetConfig.IMAGE_SIZE,
        ds_mean=DatasetConfig.MEAN,
        ds_std=DatasetConfig.STD,
        batch_size=TrainingConfig.BATCH_SIZE,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Creating ModelCheckpoint callback. 
    # We'll save the model on basis on validation f1-score.
    model_checkpoint = ModelCheckpoint(
        monitor="valid/f1",
        mode="max",
        filename="ckpt_{epoch:03d}-vloss_{valid/loss:.4f}_vf1_{valid/f1:.4f}",
        auto_insert_metric_name=False,
    )

    # Creating a learning rate monitor callback which will be plotted/added in the default logger.
    lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

                               # Initializing the Trainer class object.
    trainer = pl.Trainer(
        accelerator="auto",  # Auto select the best hardware accelerator available
        devices="auto",  # Auto select available devices for the accelerator (For eg. mutiple GPUs)
        strategy="auto",  # Auto select the distributed training strategy.
        max_epochs=TrainingConfig.NUM_EPOCHS,  # Maximum number of epoch to train for.
        enable_model_summary=False,  # Disable printing of model summary as we are using torchinfo.
        callbacks=[model_checkpoint, lr_rate_monitor],  # Declaring callbacks to use.
        precision="16-mixed",  # Using Mixed Precision training.
    )

    # Start training
    trainer.fit(model, data_module)
