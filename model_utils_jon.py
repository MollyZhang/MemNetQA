import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import re
import os
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import DistilBertModel


class SimpleDistilBERT(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.attn = nn.Linear(self.model.config.dim,self.model.config.dim).to(self.device)
        self.linear = nn.Linear(self.model.config.dim,
                                self.model.config.num_labels).to(self.device)
        self.dropout = nn.Dropout(self.model.config.qa_dropout).to(self.device)

    def forward(self, batch):
        inputs, starts, ends = self.prepare_data(batch)
        hidden = self.model(**inputs)[0] # batch_size,seq_len,768
        attn_weights = F.softmax(self.attn(hidden),dim=1) # batch_size,seq_len,768
        # attn_weights = torch.transpose(attn_weights,1,2)
        # print(f'hidden size {hidden.size()}, attn size {attn_weights.size()}')
        hidden = hidden*attn_weights #attn applied
        logits = self.linear(self.dropout(hidden))
        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(start_logits, starts) + loss_fct(end_logits, ends)

        pred_answers = {}
        for i in range(batch.batch_size):
            s = torch.argmax(start_logits[i]).item()
            e = torch.argmax(end_logits[i]).item()
            pred_answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i, s:e+1]))
            pred_answers[batch.id[i]] = pred_answer
        return pred_answers, loss, logits

    def prepare_data(self, batch):
        text_pairs = [(batch.context, q) for q in batch.q]
        max_len = max([len(" ".join([c, q]).split(" ")) for (c, q) in text_pairs])
        if max_len < 400:
            max_len = None
        inputs = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_pairs,
            add_special_tokens=True, return_token_type_ids=False,
            pad_to_max_length=True, return_tensors="pt",
            max_length=max_len)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        # find start and end position in tokenized input
        max_len = inputs["input_ids"].shape[1]
        starts = []
        ends = []
        for a, a_i in zip(batch.a, batch.a_index):
            start = len(self.tokenizer.tokenize(batch.context[:a_i])) + 1
            end = len(self.tokenizer.tokenize(batch.context[:a_i + len(a)]))
            starts.append(start)
            ends.append(end)
        starts = torch.tensor(starts).to(self.device)
        ends = torch.tensor(ends).to(self.device)
        return inputs, starts, ends
