import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import re
import os
from transformers import BertTokenizer, BertForSequenceClassification



class BaseModel(nn.Module):
    """ use trainable word embedding to directly predict labels """
    def __init__(self, info, pretrain=False, emb_dim=300, device="cuda"):
        super().__init__()
        self.info = info
        self.device = device
        self.prep_embedding(pretrain, emb_dim)
        self.linear_layer = nn.Linear(emb_dim, len(info["labels"])) 

    def prep_embedding(self, pretrain, emb_dim):
        if pretrain:
            emb_matrix = self.load_pretrained_embedding(pretrain)
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else:
            vocab_size = len(self.info["vocab"])
            self.embedding = nn.Embedding(vocab_size, emb_dim)


    def forward(self, batch):
        emb = self.embedding(batch.text)
        preds = self.linear_layer(emb)
        return preds

    def load_pretrained_embedding(self, emb_name):
        fname = "./data/pretrained_embedding/{}_vocab.vec".format(emb_name)
        df = pd.read_csv(fname, names=["word", "vec"])    
        w2v = dict(zip(df.word, df.vec.apply(lambda x: [float(i) for i in x.split(" ")])))
        emb_matrix = []
        for word in self.info["vocab"]:
            if word in w2v:
                emb_matrix.append(w2v[word])
            else:
                vec = [np.random.normal() for i in range(300)]
                emb_matrix.append(vec)
        emb_matrix = torch.tensor(emb_matrix).to(self.device)
        return emb_matrix


class GRU(BaseModel):
    def __init__(self, info, pretrain=False, device="cuda",
                 emb_dim=300, hidden_unit=200, num_layer=1, direction=2):
        super().__init__(info, pretrain=pretrain, device=device, emb_dim=emb_dim)
        self.direction = direction
        self.num_layer=num_layer
        self.hidden_unit = hidden_unit
        self.grucell = nn.GRUCell(emb_dim, hidden_unit)
        self.grucell_back = nn.GRUCell(emb_dim, hidden_unit)
        self.linear_layer = nn.Linear(self.direction * hidden_unit, 
                                      len(self.info["labels"]))
        
    def forward(self, batch):
        emb = self.embedding(batch.text)
        seq_len = emb.shape[0]
        
        # one direction
        h0 = torch.rand(batch.batch_size, self.hidden_unit, 
            requires_grad=True).to(self.device)
        h = (h0 - 0.5)/self.hidden_unit
        out = []
        for i in range(seq_len):
            h = self.grucell(emb[i, :].reshape(1, -1), h)
            out.append(h)
        out = torch.stack(out, dim=1).squeeze(dim=0)
        
        # another direction
        if self.direction > 1:
            h0 = torch.rand(batch.batch_size, self.hidden_unit, 
                requires_grad=True).to(self.device)
            h = (h0 - 0.5)/self.hidden_unit
            out_backward = []
            for i in range(seq_len-1, -1, -1):
                h = self.grucell_back(emb[i, :].reshape(1, -1), h)
                out_backward = [h] + out_backward
            out_backward = torch.stack(out_backward, dim=1).squeeze(dim=0)
            out = torch.cat((out, out_backward), dim=1)
        preds = self.linear_layer(out)
        return preds




