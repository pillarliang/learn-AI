from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BertPreTrainedModel,
    BertModel,
    AutoConfig,
    AdamW,
    get_scheduler,
)
import json
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer(checkpoint)

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 1e-5
batch_size = 4
epoch_num = 3


class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


train_data = AFQMC(f"{current_dir}/data/AFQMC/train.json")
test_data = AFQMC(f"{current_dir}/data/AFQMC/test.json")
valid_data = AFQMC(f"{current_dir}/data/AFQMC/valid.json")


def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2, batch_label = [], [], []
    for sample in batch_samples:
        batch_sentence_1.append(sample["sentence1"])
        batch_sentence_2.append(sample["sentence2"])
        batch_label.append(sample["label"])

    X = tokenizer(
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    y = torch.tensor(batch_label)
    return X, y


train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn
)
valid_dataloader = DataLoader(
    valid_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn
)


# model
class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, X):
        outputs = self.bert(**X)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits


config = AutoConfig.from_pretrained(checkpoint)
model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config)


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    return total_loss


def test_loop(dataloader, model, mode="Test"):
    assert mode in ["Test", "Valid"]
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:  # 每次循环取出一个batch_size
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size

    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct


loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader),
)

total_loss = 0.0
best_acc = 0.0

for t in range(epoch_num):
    total_loss = train_loop(
        train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss
    )
    valid_acc = test_loop(valid_dataloader, model, mode="Valid")

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(
            model.state_dict(),
            f"epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin",
        )
