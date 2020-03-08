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
        self.linear = nn.Linear(self.model.config.dim, 
                                self.model.config.num_labels).to(self.device)
        self.dropout = nn.Dropout(self.model.config.qa_dropout).to(self.device)

    def forward(self, batch):
        inputs, starts, ends = self.prepare_data(batch)
        hidden = self.model(**inputs)[0]
        logits = self.linear(self.dropout(hidden))
        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(start_logits, starts) + loss_fct(end_logits, ends)

        pred_answers = {}
        for i in range(batch.batch_size):
            try:
                s = torch.argmax(start_logits[i]).item()
                e = torch.argmax(end_logits[i]).item()
            except:
                print(i, start_logits.shape, end_logits.shape)
                raise
            pred_answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i, s:e+1]))
            pred_answers[batch.id[i]] = pred_answer
        return pred_answers, loss, logits

    def prepare_data(self, batch):
        text_pairs = [(batch.context, q) for q in batch.q] 
        max_len = max([len(" ".join([c, q]).split(" ")) for (c, q) in text_pairs])
        if max_len < 340:
            max_len = None
        else:
            max_len = 512        
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
            if start > 512:
                start, end = 0, 0
            starts.append(start)
            ends.append(end)
        starts = torch.tensor(starts).to(self.device)
        ends = torch.tensor(ends).to(self.device)
        return inputs, starts, ends


class AlBertQA(nn.Module):
    def __init__(self, albert_name="ALBERT-base", device="cuda"):
        super().__init__()
        if albert_name == "ALBERT-base":
            albert_base_configuration = AlbertConfig(
                hidden_size=768, num_attention_heads=12, intermediate_size=3072)
        elif albert_name == "ALBERT-xxlarge":
            albert_base_configuration = AlbertConfig()
        else:
            raise
