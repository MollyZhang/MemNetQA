from torchtext.data import TabularDataset, Field, RawField, BucketIterator, Iterator
import torch
import numpy as np
import re
import os
import pickle


def prep_all_data(batch_size=32, path="data", 
                  device="cuda", bert=True,  
                  train_file = "train_real.csv",
                  val_file = "holdout.csv"):
    
    tokenize = lambda x: re.split(" ", x.lower())
    text_field = Field(sequential=True, tokenize=tokenize, 
                       lower=True, include_lengths=True)
    raw_text_field = RawField()
    label2idx = pickle.load(open("./data/label2idx.pkl", "rb"))
    label_proc = lambda x: torch.tensor([label2idx[i] for i in x.split(" ")])
    label_field = RawField(preprocessing=label_proc, is_target=True)

    trn, vld = TabularDataset.splits(path=path, 
        train=train_file, validation=val_file,
        format='csv', skip_header=True,
        fields=[("utterances", text_field),
                #("utterances", raw_text_field), 
                ("IOB Slot tags", label_field)])

    text_field.build_vocab(trn)
    train_iter, val_iter = BucketIterator.splits(
        (trn, vld),
        batch_sizes=(batch_size, batch_size),
        device=torch.device(device), 
        sort_key=lambda x: len(x.utterances), 
        sort_within_batch=False,
        repeat=False)

    train_data = BaseWrapper(train_iter, text_field=text_field, path=path, 
        sample_size=len(trn), batch_size=batch_size)
    val_data = BaseWrapper(val_iter, text_field=text_field, path=path,
        sample_size=len(vld), batch_size=batch_size)
    return train_data, val_data, trn, vld, train_iter, val_iter



class BaseWrapper(object):
    def __init__(self, data, text_field=None, path="data", device="cuda",
                 sample_size=None, batch_size=None):
        self.data = data
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.text_field = text_field
        self.device = device

    def __iter__(self):
        for batch in self.data:
            x = batch.utterances[0]
            y = torch.stack(batch.__dict__["IOB Slot tags"], axis=0)
            yield x, y
    
    def __len__(self):
        return self.sample_size

