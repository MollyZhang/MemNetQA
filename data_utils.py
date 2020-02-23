import torch
import numpy as np
import pandas as pd
import re
import os
import pickle
import itertools


def prep_all_data(file_dict, batch_size=1, device="cuda", save=False):
    df_train = pd.read_csv(file_dict["train"])
    df_val = pd.read_csv(file_dict["val_text"], names=["utterances"]).join(
        pd.read_csv(file_dict["val_label"], names=["IOB Slot tags"]))

    # build vocab from train
    tokenize = lambda x: re.split(" ", x.lower())
    vocab = sorted(list(set(list(itertools.chain.from_iterable(    
    list(df_train["utterances"].apply(tokenize)))))))
    vocab = ["<pad>", "<unk>"] + vocab
    text2idx = {text: i for i, text in enumerate(vocab)}
    idx2text = {i: text for i, text in enumerate(vocab)}
    
    # build label from train
    labels = sorted(list(set(" ".join(df_train["IOB Slot tags"]).split(" "))))
    labels = ["<UNK>"] + labels
    label2idx = {label: i for i, label in enumerate(labels)}
    idx2label = {i: label for i, label in enumerate(labels)}
    
    if save:
        np.save("./data/vocab.npy", np.array(vocab))
        np.save("./data/labels.npy", np.array(labels))

    # build batch
    if batch_size != 1:
        raise(Exception("not implemented"))

    train_data = Batch(df_train, text2idx, label2idx, device=device)
    val_data = Batch(df_val, text2idx, label2idx, device=device)
    
    extra = {"vocab": vocab, "labels": labels}
    return train_data, val_data, extra


class Batch(object):
    def __init__(self, df, text2idx, label2idx, device="cuda"):
        self.df = df
        self.t2i = text2idx
        self.l2i = label2idx
        self.device = device

    def __iter__(self):
        idx = np.random.permutation(self.df.index)    
        for i in idx:
            sample = Sample(i, self.df.loc[i], self.t2i, self.l2i, device=self.device)
            yield sample

    def __len__(self):
        return self.df.shape[0]



class Sample(object):
    def __init__(self, sample_id, data, text2idx, label2idx, device="cuda"):
        self.raw_text = data.utterances.lower().split(" ")
        self.raw_label = data["IOB Slot tags"].split(" ")
        assert(len(self.raw_text) == len(self.raw_label))
        self.t2i = text2idx
        self.l2i = label2idx
        self.sample_id = sample_id
        self.text = torch.tensor(self.convert_text()).to(device)
        self.label = torch.tensor(self.convert_label()).to(device)
        self.batch_size = 1

    def convert_text(self):
        self.text = []
        for t in self.raw_text:
            try:
                self.text.append(self.t2i[t])
            except KeyError:
                self.text.append(self.t2i["<unk>"])
        return self.text

    def convert_label(self):
        self.label = []
        for l in self.raw_label:
            try:
                self.label.append(self.l2i[l])
            except KeyError:
                self.label.append(self.l2i["<UNK>"])
        return self.label

