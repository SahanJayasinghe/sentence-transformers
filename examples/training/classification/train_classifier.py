"""
The system trains Sentence Emeddings using BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on a categorical dataset
with classification (softmax) loss function.

Usage:
python train_classifier.py

OR
python train_classifier.py pretrained_transformer_model_name
"""

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
import logging
from datetime import datetime
import sys
import os
# import gzip
# import csv
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset_path = 'data/SigmaLaw-ABSA.csv'

if not os.path.exists(dataset_path):
    raise FileNotFoundError("CSV file containing the Dataset is not found at: {}".format(dataset_path))

logging.info("Reading the dataset from {}".format(dataset_path))
valid_labels = [0, 1, 2]
label_dict = {'-1': 0, '0': 1, '1': 2}
# label_dict = {'0': 0, '2': 1, '3': 2, '5': 3}
train_split, val_split = 0.8, 0.1
train_batch_size = 16

df = pd.read_csv(dataset_path)
num_samples = df.shape[0]
train_samples_limit = int(train_split * num_samples)
val_samples_limit = train_samples_limit + int(val_split * num_samples)

train_samples, val_samples = [], []

for index, row in df.iterrows():
    if index >= val_samples_limit: break

    sentence = row['Sentence'].strip()
    sentiment = str(row['Overall Sentiment'])   # Sentiment: -1 (negative) | 0 (neutral) | 1 (positive)
    if sentiment not in label_dict.keys(): continue
    label = label_dict[sentiment]
    # label = int(row['Overall Sentiment']) + 1
    # if label not in valid_labels: continue

    if index < train_samples_limit:
        train_samples.append(InputExample(texts=[sentence], label=label))
    else:
        val_samples.append(InputExample(texts=[sentence], label=label))


# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
dropout_rate = None

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

modules_list = [word_embedding_model, pooling_model]

if dropout_rate != None:
    modules_list.append(models.Dropout(dropout=dropout_rate))

model = SentenceTransformer(modules=modules_list)


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.ClassificationLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=len(label_dict)
)

val_dataloader = DataLoader(val_samples, shuffle=False, batch_size=train_batch_size)
dev_evaluator = LabelAccuracyEvaluator(val_dataloader, softmax_model=train_loss, name='validation-set')


# Configure the training
num_epochs = 3
eval_steps = int(len(train_dataloader)/4)
checkpoint_path = 'output/checkpoints'
checkpoint_save_steps = eval_steps
checkpoint_limit = 20

model_save_path = 'output/classifier_' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=eval_steps,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    checkpoint_path=checkpoint_path,
    checkpoint_save_steps=checkpoint_save_steps,
    checkpoint_save_total_limit=checkpoint_limit
)
