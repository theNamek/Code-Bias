import json
import torch
from torch.utils.data import Dataset
from utils import *
import os
import jsonlines

class CodeDataset(Dataset):
    def __init__(self, file_path="", code_model="codegen", split="train"):
        data_path = base_data_path if file_path == "" else file_path
        data_path = os.path.join(data_path, code_model, f"{split}.jsonl")

        self.data = []
        with open(data_path, 'r+', encoding="utf-8") as f:
            for line in jsonlines.Reader(f):
                self.data.append(line)

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def __getitem__(self, index):
        json_data = self.data[index]
        code = json_data['code']
        label = json_data['label']
        return code, int(label)

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = CodeDataset('train.jsonl', code_model="codegen", split="train")
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(f"Data length: {len(train_data_loader)}")
    data = next(iter(train_data_loader))
    print(data)