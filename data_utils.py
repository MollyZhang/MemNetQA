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
    df_test = pd.read_csv(file_dict["test_text"], names=["utterances"])
    df_test["IOB Slot tags"] = "dummy"

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
    if batch_size == 1:
        train_data = Batch(df_train, text2idx, label2idx, device=device)
        val_data = Batch(df_val, text2idx, label2idx, device=device)
        test_data = Batch(df_test, text2idx, label2idx, device=device)
    elif batch_size == -1: # variable batch size
        train_data = VarBatch(df_train, text2idx, label2idx, device=device)
        val_data = VarBatch(df_val, text2idx, label2idx, device=device)
        test_data = VarBatch(df_test, text2idx, label2idx, device=device)
    else: 
        raise(Exception("not implemented"))
    extra = {"vocab": vocab, "labels": labels}
    return train_data, val_data, test_data, extra


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
            yield SampleBatch([sample], device=self.device)

    def __len__(self):
        return self.df.shape[0]


class VarBatch(object):
    """ variable batch size that bunch text of same length together """
    def __init__(self, df, text2idx, label2idx, max_batch_size=32, device="cuda"):
        self.df = df
        self.t2i = text2idx
        self.l2i = label2idx
        self.device = device
        self.mbs = max_batch_size
        self.df["len"] = self.df.utterances.apply(lambda x:len(x.split(" ")))
        self.df = self.df.sort_values(by="len").reset_index()
        self.pointer = 0
        self.num_samples = self.df.shape[0]

    def __iter__(self):
        while self.pointer < self.num_samples:
            samples = []
            for i in range(self.mbs):
                sample = Sample(self.df.iloc[self.pointer+i]["index"],
                    self.df.iloc[self.pointer+i], self.t2i, self.l2i, device=self.device)
                samples.append(sample)
                if self.pointer + i + 1 >= self.num_samples:
                    break
                elif self.df.iloc[self.pointer+i+1]["len"] > self.df.iloc[self.pointer+i]["len"]:
                    break
            self.pointer += len(samples)
            yield SampleBatch(samples, device=self.device)
        self.pointer = 0

    def __len__(self):
        return self.df.shape[0]


class SampleBatch(object):
    def __init__(self, samples, device="cuda"):
        self.batch_size = len(samples)
        self.raw_text = [i.raw_text for i in samples]
        self.raw_label = [i.raw_label for i in samples]
        self.sample_id = [i.sample_id for i in samples]
        self.text = torch.stack([i.text for i in samples], dim=0).to(device)
        self.label = torch.stack([i.label for i in samples], dim=0).to(device)
        self.seq_len = self.text.shape[1]

class Sample(object):
    def __init__(self, sample_id, data, text2idx, label2idx, device="cuda"):
        self.raw_text = data.utterances.lower().split(" ")
        self.raw_label = data["IOB Slot tags"].split(" ")
        if self.raw_label[0] != "dummy":
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

