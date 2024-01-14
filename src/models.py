import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, config, num_vocab, tokenizer, embed_dim=768, hidden_dim=100, pretrained=False, weight=None):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, embed_dim)
        self.model = nn.LSTM(
            input_size=embed_dim,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.tokenizer = tokenizer

        if pretrained is True:
            self.embedding.weight = weight
        
    def forward(self, x, debug=False):
        """
            return: [2 * hidden_dim]
        """
        x = self.tokenizer(x)["input_ids"] # [B, L]
        x = torch.tensor(x).cuda()
        if debug:
            print(x)
        x = self.embedding(x) # [B, L, emb_dim]
        x, _ = self.model(x) # [B, L, 2 * hidden_dim]
        x = x[:, -1, :]
        if debug:
            print(x)

        return x

class BERT(nn.Module):
    def __init__(self, BERT_model, tokenizer):
        super().__init__()
        self.model = BERT_model
        self.tokenizer = tokenizer

    def forward(self, x, debug=False):
        x = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt")
        if debug:
            print(x)
        x = {k: v.cuda() for k, v in x.items()}
        x = self.model(**x).pooler_output
        if debug:
            print(x)
        
        return x
        