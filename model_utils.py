import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import re
import os
from transformers import AlbertConfig, AlbertModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class PreTrainedSQuAD(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")
        self.device = device

    def forward(self, batch):
        answer_dict = {}
        for q, q_id in zip(batch.q, batch.id):
            inputs = self.tokenizer.encode_plus(q, batch.context, 
                add_special_tokens=True, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"].tolist()[0]
            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = self.model(**inputs)
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            answer_dict[q_id] = answer
        return answer_dict



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


