from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
import logging
from ..util import batch_to_device
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
# from torchmetrics.functional import accuracy, f1
# import csv

logger = logging.getLogger(__name__)

class ClassificationTaskEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy and F1 scores on a labeled dataset
    This requires a model with LossFunction.ClassificationLoss
    """

    def __init__(self, dataloader: DataLoader, classification_layer: Module, num_classes: int):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.classification_layer = classification_layer
        self.num_classes = num_classes

    def __call__(self, model):
        pred_result = np.empty((0,), dtype=np.int32)
        labels = np.empty((0,), dtype=np.int32)

        model.eval()
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            labels = np.append(labels, label_ids.to('cpu').numpy(), axis=0)
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.classification_layer(features, labels=None)

            pred_result = np.append(pred_result, torch.argmax(prediction, dim=1).to('cpu').numpy(), axis=0)

        acc = accuracy_score(labels, pred_result)
        f1_micro = f1_score(labels, pred_result, average='micro')
        f1_macro = f1_score(labels, pred_result, average='macro')

        # pred_result_tensor = torch.tensor(pred_result)
        # labels_tensor = torch.tensor(labels)

        # acc = accuracy(pred_result_tensor, labels_tensor)
        # f1_micro = f1(pred_result_tensor, labels_tensor, average='micro')
        # f1_macro = f1(pred_result_tensor, labels_tensor, average='macro', num_classes=self.num_classes)

        return {'accuracy': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
