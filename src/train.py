import torch
import torch.nn as nn
from classifier import BiasScoreClassifier
import torch.optim as optim
from dataset import CodeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import *
import torch, datetime

set_seed(1)

class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'a+')

    def log(self, msg):
        formatted = f'[{datetime.datetime.now().replace(microsecond=0).isoformat()}] {msg}'
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()

def collate(x):
    return [i[0] for i in x], torch.tensor([i[1] for i in x]).view(-1, 1).float()

def train_one_epoch(model, dataloader, optimizer, loss_fn):
    losses = []
    correct, total = [], []
    for idx, data in enumerate(tqdm(dataloader)):
        code, label = data
        label = label.cuda()
        output = model(code)
        # print(output, label)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = (output >= 0.5).int()
        losses.append(loss.item())
        correct.append(prediction.eq(label).sum().item())
        total.append(label.numel())

    return {
        "loss": np.mean(losses),
        "acc": sum(correct) / sum(total)
    }

def test(model, dataloader, loss_fn):
    losses = []
    correct = []
    total = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            code, label = data
            label = label.cuda()
            output = model(code)
            loss = loss_fn(output, label)
 
            losses.append(loss.item())
            prediction = (output >= 0.5).int()
            correct.append(prediction.eq(label).sum().item())
            total.append(label.numel())

    return {
        "loss": np.mean(losses),
        "acc": sum(correct) / sum(total)
    }
    

def _get_optimizer(model: BiasScoreClassifier, config):
    optimizer = optim.Adam([
            {"params": model.lm.parameters(), "lr": config["bert_lr"]},
            {"params": model.classifier.parameters(), "lr": config["classifier_lr"]}
        ],
    )

    return optimizer

def train(logger, config):
    best_model, best_loss, best_acc = None, 1., 0.
    model = BiasScoreClassifier(config).cuda()

    optimizer = _get_optimizer(model, config)
    # train_dataset = CodeDataset(code_model=config["code_model"], split="train")
    datasets = {
        s: CodeDataset(code_model=config["code_model"], split=s) for s in ["train", "test", "val"]
    }

    dataloaders = {
        k: DataLoader(
        d,
        batch_size=config["batch_size"] if k == "train" else 1,
        shuffle=True,
        collate_fn=collate,) for k, d in datasets.items()
    } 

    loss_fn = nn.BCELoss()

    for epoch_idx in range(config["epochs"]):
        # print(f"----------- Epoch {epoch_idx + 1} ----------")
        logger.log(f"----------- Epoch {epoch_idx + 1} ----------")

        train_metric = train_one_epoch(model, dataloaders["train"], optimizer, loss_fn)
        # print(f"[Train] loss: {train_metric['loss']}, acc: {train_metric['acc']}")
        logger.log(f"[Train] loss: {train_metric['loss']}, acc: {train_metric['acc']}")

        for split in ["test", "val"]:
            metric = test(model, dataloaders[split], loss_fn)
            # print(f"[{split}] loss: {metric['loss']}, acc: {metric['acc']}")
            logger.log(f"[{split}] loss: {metric['loss']}, acc: {metric['acc']}")

            if split == "val" and metric['loss'] < best_loss:
                best_loss = metric["loss"]
                save_folder = os.path.join(project_path, "saved", config["code_model"])
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                torch.save(model.state_dict(), os.path.join(save_folder, "best.pt"))
                # print(f"Best model saved with accuracy {metric['acc']}")
                logger.log(f"Best model saved with accuracy {metric['acc']}")

def test_only(logger, config):
    model = BiasScoreClassifier(config).cuda()
    model_path = os.path.join(project_path, "saved", config["code_model"], "best.pt")
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist")
        return 

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    if "test_data_path" in config:
        file_path = config["test_data_path"]
    else:
        file_path = ""

    datasets = {
        s: CodeDataset(code_model=config["code_model"], split=s, file_path=file_path) for s in ["test"]
    }

    dataloaders = {
        k: DataLoader(
        d,
        batch_size=config["batch_size"] if k == "train" else 1,
        shuffle=True,
        collate_fn=collate,) for k, d in datasets.items()
    } 
    loss_fn = nn.BCELoss()

    split = "test"
    metric = test(model, dataloaders[split], loss_fn)
    # print(f"[{split}] loss: {metric['loss']}, acc: {metric['acc']}")
    logger.log(f"[{split}] loss: {metric['loss']}, acc: {metric['acc']}")

    return 

def main():
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--eval", action="store_true", default=False)
    args = parser.parse_args() 
    config = load_config(os.path.join(base_config_path, f"{args.config}.yml"))
    logdir = config["logdir"]
    reopen_to_flush = True
    logger = Logger(os.path.join(logdir, 'log.txt'), reopen_to_flush)
    logger.log(f'Logging to {logdir}')
    logger.log(f"Config: {config}")
    # print(f"Config: {config}")

    if not args.eval:
        train(logger, config)
    else:
        test_only(logger, config)

if __name__ == "__main__":
    main()
    