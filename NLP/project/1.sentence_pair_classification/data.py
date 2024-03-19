from torch.utils.data import Dataset, DataLoader
import torch
import json


class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data


def get_dataLoader(datasets, tokenizer, batch_size=None, shuffle=False):

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

        return {"batch_inputs": X, "batch_labels": y}

    return DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, collate_fn=collote_fn
    )
