import torch
import numpy as np
import pandas as pd
import re
import os
import pickle
import itertools
import json


def prep_data(json_name, batch_size="paragraph", device="cuda"):
    data = json.load(open(json_name, "r"))["data"]
    data_iter = ParaBatch(data)
    return data_iter

class ParaBatch(object):
    """  a batch bundles qa from same paragraph together """
    def __init__(self, data, device="cuda"):
        self.device = device
        self.num_title = len(data)
        self.data = data
        self.expand_paragraph()

    def expand_paragraph(self):
        self.paragraph = []
        for i in self.data:
            self.paragraph.extend(i["paragraphs"])

    def __iter__(self):
        for p in self.paragraph:
            yield Paragraph(p)

    def __len__(self):
        return len(self.paragraph)


class Paragraph(object):
    def __init__(self, p, device="cuda"):
        self.context = p["context"]
        self.expand_qas(p)
        self.num_qa = len(p)

    def expand_qas(self, p):
        self.q = []
        self.a = []
        self.a_index = []
        self.id = []
        self.impossible = []
        for qa in p["qas"]:
            self.q.append(qa["question"])
            self.a.append(qa["answers"][0]["text"]) 
            self.a_index.append(qa["answers"][0]["answer_start"])
            self.id.append(qa["id"])
            self.impossible.append(qa["is_impossible"])
