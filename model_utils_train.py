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
        loss_fct = nn.BCEWithLogitsLoss(reduction="sum")
        loss = loss_fct(start_logits, starts) + loss_fct(end_logits, ends)
        
        pred_answers = {}
        for i in range(batch.batch_size):
            s = torch.argmax(start_logits[i]).item()
            e = torch.argmax(end_logits[i]).item()
            pred_answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i, s:e+1]))
            pred_answers[batch.id[i]] = pred_answer
        
        return pred_answers, loss


    def prepare_data(self, batch):
        text_pairs = [(batch.context, q) for q in batch.q] 
        inputs = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_pairs,
            add_special_tokens=True, return_token_type_ids=False,
            pad_to_max_length=True, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        # find start and end position in tokenized input
        max_len = inputs["input_ids"].shape[1]
        starts = []
        ends = []
        for a, a_i in zip(batch.a, batch.a_index):
            start = len(self.tokenizer.tokenize(batch.context[:a_i])) + 1
            end = len(self.tokenizer.tokenize(batch.context[:a_i + len(a)]))
            one_hot_start, one_hot_end = torch.zeros(max_len), torch.zeros(max_len)
            one_hot_start[start] = 1
            one_hot_end[end] = 1
            starts.append(one_hot_start)
            ends.append(one_hot_end)
        starts = torch.stack(starts).float().to(self.device)
        ends = torch.stack(ends).float().to(self.device)
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


