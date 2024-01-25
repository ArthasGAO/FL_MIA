# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 06:03:43 2023

@author: yuanzhe

"""
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Metric


class MLP(L.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()

        self.learning_rate = learning_rate
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
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
        }, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "scores": y_pred, "y": labels}

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
        }, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "scores": y_pred, "y": labels}

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
        }, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "scores": y_pred, "y": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
