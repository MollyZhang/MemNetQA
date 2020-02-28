import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import re
import os
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from transformers import BertForTokenClassification



class BertSlot(nn.Module):
    def __init__(self, info,  
                 bert_name="bert-base-uncased", device="cuda"):
        super().__init__()
        self.info = info
        self.output_dim = len(self.info["labels"])
        if bert_name == "bert-base-uncased":
            emb_dim = 768
        else:
            raise
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.bert = BertModel.from_pretrained(bert_name)
        self.grucell = nn.GRUCell(self.output_dim, emb_dim)
        self.fc = nn.Linear(emb_dim, self.output_dim)
        self.sm = nn.Softmax(dim=1)
        self.device = device

    def forward(self, batch):
        input_ids = self.prepare_data(batch)
        outputs = self.bert(input_ids)
        label_output = torch.rand(batch.batch_size, self.output_dim).to(self.device)
        out = []
        h = torch.mean(outputs[0], dim=1)       
        for i in range(batch.seq_len):
            h = self.grucell(label_output, h)
            logits = self.fc(h)
            label_output = self.sm(logits)
            out.append(logits)
        out = torch.stack(out, dim=1)
        return out


    def prepare_data(self, batch):
        texts = []
        for i in batch.raw_text:
            text = "[CLS] " + " ".join(i) + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(text)
            text_index = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            texts.append(text_index)
        return torch.tensor(texts).to(self.device)



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
        seq_len = emb.shape[1]
        # one direction
        h0 = torch.rand(batch.batch_size, self.hidden_unit, 
            requires_grad=True).to(self.device)
        h = (h0 - 0.5)/self.hidden_unit
        out = []
        for i in range(seq_len):
            h = self.grucell(emb[:, i, :], h)
            out.append(h)
        out = torch.stack(out, dim=1)
        
        # another direction
        if self.direction > 1:
            h0 = torch.rand(batch.batch_size, self.hidden_unit, 
                requires_grad=True).to(self.device)
            h = (h0 - 0.5)/self.hidden_unit
            out_backward = []
            for i in range(seq_len-1, -1, -1):
                h = self.grucell_back(emb[:, i, :], h)
                out_backward = [h] + out_backward
            out_backward = torch.stack(out_backward, dim=1)
            out = torch.cat((out, out_backward), dim=2)
        preds = self.linear_layer(out)
        return preds



