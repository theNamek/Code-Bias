import torch
import transformers

import torch.nn as nn
from utils import *
import os

class BiasScoreClassifier(nn.Module):
    def __init__(self, config):
        super(BiasScoreClassifier, self).__init__()
        self.classifier = _get_classifier(config)
        self.lm = _get_lm(config)

    def forward(self, x, debug=False):
        x = self.lm(x, debug=debug)
        x = self.classifier(x)
        if debug:
            print(x)
        
        return x

def _get_classifier(config):
    """
        : Return: lm_head (MLP | Transformer)
    """
    # model = [nn.Linear(768, 2)]
    # model.append(nn.ReLU())

    if "lm" in config and config["lm"] == "lstm":
        model = nn.Sequential(
            nn.Linear(2 * config["hidden_dim"], 1),
            nn.Sigmoid()
        )
    else:
        model = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    return model

def _get_lm(config):
    """
        :Return: tokenizer, model
    """
    from transformers import AutoTokenizer
    from transformers import BertModel
    import models

    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if "lm" in config and config["lm"] == "lstm":
        weight = model.embeddings.word_embeddings.weight
        model = models.LSTM(config, num_vocab=weight.shape[0], embed_dim=weight.shape[1], tokenizer=tokenizer, hidden_dim=config["hidden_dim"], pretrained=bool(config["pretrained"]), weight=weight)
    else:
        model = models.BERT(model, tokenizer)
    
    print("Language model arch: ", model)
    return model


if __name__ == "__main__":
    # _get_lm(None)
    # _get_classifier(None)

    sample = ["def find_lonely_people(people, ethnicity):\n    lonely_people = []\n    for person in people:\n        if person[ethnicity] == 'lonely':\n            lonely_people.append(person)\n    return lonely_people"]

    config = load_config(os.path.join(base_config_path, f"base.yml"))
    model = BiasScoreClassifier(config)
    output = model(sample, debug=True)

