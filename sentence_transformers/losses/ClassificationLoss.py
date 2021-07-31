import os
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from copy import deepcopy
from ..SentenceTransformer import SentenceTransformer
import logging


logger = logging.getLogger(__name__)

class ClassificationLoss(nn.Module):
    """
    This loss is used to measure the classifcation loss of a single sentence against given classes.
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_classes: int,
                 use_custom_loss: bool = False,
                 opposite_class_loss_weight: float = 2.0):
        super(ClassificationLoss, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.mid_class = int((num_classes + 1) / 2)
        self.use_custom_loss = use_custom_loss
        self.opposite_class_loss_weight = opposite_class_loss_weight
        self.classifier = nn.Linear(sentence_embedding_dimension, num_classes)
        logger.info("ClassificationLoss Linear Model input dim: {} | output dim: {}".format(sentence_embedding_dimension, num_classes))

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = self.model(sentence_features[0])['sentence_embedding']

        output = self.classifier(reps)
        if self.use_custom_loss:
            loss_fct = self.class_loss
        else:
            loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            # loss = class_loss(output, labels.view(-1))
            return loss
        else:
            return reps, output

    def class_loss(self, pred_result: Tensor, labels: Tensor, margin=1e-20):
        """
        pred_result shape: (batch_size, num_classes) or (num_samples, num_classes)
        labels shape: (batch_size)
        margin: float value to add to the `torch.log` function input to prevent log(0)
        """
        
        softmax_fn = nn.Softmax(dim=1)
        s_out = softmax_fn(pred_result)
        # pred_cls = torch.argmax(s_out, dim=1)
        
        cls_weights = torch.empty((pred_result.shape[0], self.num_classes), dtype=torch.float32, device='cuda')

        for b in range(pred_result.shape[0]):
            for c in range(self.num_classes):
                if labels[b] == c:
                    cls_weights[b][c] = 0
                elif labels[b] < self.mid_class:
                    # cls_weights[b][c] = int(c >= self.mid_class) + 1
                    cls_weights[b][c] = self.opposite_class_loss_weight if c >= self.mid_class else 1
                else:
                    # cls_weights[b][c] = int(c < self.mid_class) + 1
                    cls_weights[b][c] = self.opposite_class_loss_weight if c < self.mid_class else 1
        
        inverse_probability_log_values = torch.log(torch.add(torch.sub(
            torch.ones(pred_result.shape[0], self.num_classes, dtype=torch.float32, device='cuda'),
            s_out
        ), margin))

        loss_matrix = torch.mul(inverse_probability_log_values, cls_weights)

        loss = torch.mean(torch.sum(loss_matrix, 1)) * (-1)
        loss.requires_grad_(True)
        return loss
    
    def set_best_state(self):
        """
        store the state of the last layer parameters: ['classifier.weight', 'classifier.bias']
        in best_state class attribute
        """
        self.best_state = {key: self.state_dict()[key] for key in list(self.state_dict().keys())[-2:]}
        # self.best_state = deepcopy(self.state_dict())

    def save_current_state(self, path: str, filename: str = 'classifier_state_dict.pt'):
        """
        save the state of the last layer (classifier) parameters: ['classifier.weight', 'classifier.bias']
        """
        os.makedirs(path, exist_ok=True)
        classifier_state = {key: self.state_dict()[key] for key in list(self.state_dict().keys())[-2:]}
        torch.save(classifier_state, os.path.join(path, filename))
    
    def save_best_state(self, path: str, filename: str = 'classifier_state_dict.pt'):
        """
        save the best_state class attribute in a .pt file
        """
        if hasattr(self, 'best_state'):
            os.makedirs(path, exist_ok=True)
            torch.save(self.best_state, os.path.join(path, filename))
    
    def load_classifier_state(self, path: str):
        """
        load the last layer (classifier) state into nn.Module state
        """
        state_dict_copy = deepcopy(self.state_dict())
        classifier_state = torch.load(path)
        for key in list(classifier_state.keys()):
            state_dict_copy[key] = classifier_state[key]
        self.load_state_dict(state_dict_copy)
