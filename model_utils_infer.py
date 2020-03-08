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
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering


class PreTrainedSQuAD(nn.Module):
    """ just for inference """
    def __init__(self, 
        model_name="bert-large-uncased-whole-word-masking-finetuned-squad", device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = device

    def forward(self, batch):
        answer_dict = {}
        for q, q_id in zip(batch.q, batch.id):
            inputs = self.tokenizer.encode_plus(q, batch.context, 
                add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = self.model(**inputs)
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            answer_dict[q_id] = answer
        return answer_dict


class DistilBERTSQuAD(nn.Module):
    """ just for inference """
    def __init__(self, 
        model_name="distilbert-base-uncased-distilled-squad", device="cuda"):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        self.device = device
        self.model = self.model.to(self.device)

    def forward(self, batch):
        answer_dict = {}
        for q, q_id in zip(batch.q, batch.id):
            input_text = "[CLS] " + q + " [SEP] " + batch.context + " [SEP]"
            input_ids = torch.tensor(self.tokenizer.encode(
                input_text, max_length=512)).unsqueeze(0).to(self.device)
            answer_start_scores, answer_end_scores = self.model(input_ids)
            answer_start = torch.argmax(answer_start_scores).item()
            answer_end = torch.argmax(answer_end_scores).item() + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
            answer_dict[q_id] = answer
        return answer_dict


