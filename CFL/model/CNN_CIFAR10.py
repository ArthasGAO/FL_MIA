import torch
import lightning as pl
from torch import nn
import torchmetrics
from torch.utils.data import random_split


class CIFAR10ModelCNN(pl.LightningModule):
    def __init__(
            self,
            in_channels=3,
            out_channels=10,
            learning_rate=1e-2,
    ):
        super().__init__()

        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, out_channels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=out_channels, average='macro')
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=out_channels, average='macro')
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=out_channels, average='macro')

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _common_step(self, batch):
        images, labels = batch
        y_pred = self.forward(images)
        loss = self.loss_fn(y_pred, labels)
        return loss, y_pred, labels

    def training_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch)

        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        F1 = self.F1(y_pred, labels)
        self.log_dict({
            "train_loss": loss,
            "train_accuracy": accuracy,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1_score": F1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": y_pred, "y": labels, "train_accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch)

        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        F1 = self.F1(y_pred, labels)
        self.log_dict({
            "val_loss": loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1_score": F1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": y_pred, "y": labels, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch)
        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        F1 = self.F1(y_pred, labels)
        self.log_dict({
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": F1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": y_pred, "y": labels, "test_accuracy": accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=8.0505e-04,
                                betas=(0.851436, 0.999689),
                                amsgrad=True)
