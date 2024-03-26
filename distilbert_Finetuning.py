import json
import os
from sequence_aligner.labelset import LabelSet
from sequence_aligner.dataset import TrainingDatasetCRF
from sequence_aligner.containers import TraingingBatch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
import config
from model.DistilBERT_crf import DistilBertCrfForNer
from seqeval import metrics
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, accuracy_score
from torch import cuda
import warnings
from util.train import train_epoch, valid_epoch
from torch.optim import AdamW

# Load the custom financial dataset
raw = json.load(open('./data/FinEntity.json'))

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format.")
    except Exception as e:
        print("An error occurred:", e)

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\HP\PycharmProjects\PJ\sentiments\data\FinEntity.json"
    dataset = load_json_file(file_path)
    if dataset:
        print("JSON data loaded successfully:")
        print(dataset)

# Preprocessing stage, sequence labeling, tokenization
model_ckpt = 'distilbert/distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
label_set = LabelSet(labels=["Neutral", "Positive", "Negative"])
dataset = TrainingDatasetCRF(data=raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=128)

# Splitting training and validation data
train_size = int(config.dev_split_size * len(dataset))
validate_size = len(dataset) - train_size
train_dataset, validate_dataset = random_split(dataset, [train_size, validate_size])

train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=TraingingBatch, shuffle=True)
val_loader = DataLoader(validate_dataset, batch_size=16, collate_fn=TraingingBatch, shuffle=True)

# Model Fine-tuning phase
warnings.filterwarnings('ignore')
label_set = LabelSet(labels=["Neutral", "Positive", "Negative"])
model = DistilBertCrfForNer.from_pretrained('distilbert/distilbert-base-uncased', num_labels=len(label_set.ids_to_label.values()))
device = 'cuda:0' if cuda.is_available() else 'cpu'
model.to(device)
len_dataset = len(train_dataset)
t_total = len(train_dataset)
no_decay = ["bias", "LayerNorm.weight"]
bert_param_optimizer = list(model.distilbert.named_parameters())
crf_param_optimizer = list(model.crf.named_parameters())
optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.lr_crf},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': config.lr_crf},
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': config.crf_learning_rate},
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-6)
warmup_steps = int(t_total * config.warm_up_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

EPOCHS = config.epoch_num
for e in range(EPOCHS):
    print("=======START TRAIN EPOCHS %d=======" %(e+1))
    train_loss = train_epoch(e, model, train_loader, optimizer, scheduler, device)
    valid_epoch(e, model, val_loader, device, label_set)
