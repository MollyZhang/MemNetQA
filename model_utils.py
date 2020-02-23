import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import re
import os
from transformers import BertTokenizer, BertForSequenceClassification



class BaseModel(nn.Module):
    """ use trainable word embedding to directly predict labels """
    def __init__(self, info, ft=False, emb_dim=300):
        super().__init__()
        if ft:
            emb_matrix = torch.load("./data/emb_matrix_ft.pt")
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else:
            vocab_size = len(info["vocab"])
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        output_dim = len(info["labels"])
        self.linear_layer = nn.Linear(emb_dim, output_dim)

    def forward(self, batch):
        emb = self.embedding(batch.text)
        preds = self.linear_layer(emb)
        return preds



class GRU(nn.Module):
    def __init__(self, ft=False, device="cuda",
                 hidden_unit=200, emb_dim=300, num_layer=1, bi=False):
        super().__init__()
        if ft:
            print("use fasttext pretained embedding")
            emb_matrix = torch.load(os.path.join(path, "emb_matrix_ft.pt"))
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            print("use random initialized embedding")
            vocab = np.load("./data/vocab.npy")
            self.embedding = nn.Embedding(len(vocab), emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_unit)
        self.hidden_unit = hidden_unit
        self.device = device
        self.bi = bi
        self.num_layer=num_layer
        if self.bi:
            self.h0_bi = 2
        else:
            self.h0_bi = 1
        self.final_layer1 = nn.Linear(self.h0_bi * hidden_unit, output_dim)
        
    def forward(self, x):
        seq, raw_text = x
        x_emb, x_ngram = seq
        batch_size = len(raw_text)
        # emb shape: sequence_length, batch_size, emb_dim
        emb = self.embedding(x_emb)
        h0 = torch.rand((self.h0_bi*self.num_layer,batch_size,self.hidden_unit), 
            requires_grad=True)
        h0 = (h0 - 0.5)/self.hidden_unit
        if self.device == "cuda":
            h0 = h0.cuda()
        self.gru.flatten_parameters()
        output, h  = self.gru(emb, h0)
        preds = self.final_layer1(output[-1, :, :])
        return preds



