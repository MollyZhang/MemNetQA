import torch
import numpy as np
import pandas as pd
import re
import os
import pickle
import itertools
import json


def prep_data(json_name, version="1.1", batch_size="paragraph", device="cuda"):
    data = json.load(open(json_name, "r"))["data"]
    return ParaBatch(data, version=version)


class ParaBatch(object):
    """  a batch bundles qa from same paragraph together """
    def __init__(self, data, version="1.1", device="cuda"):
        self.device = device
        self.num_title = len(data)
        self.data = data
        self.version = version
        self.expand_paragraph()
        self.num_qa = self.get_num_qa()
        self.num_batch = len(self.paragraph)

    def expand_paragraph(self):
        self.paragraph = []
        for i in self.data:
            self.paragraph.extend(i["paragraphs"])

    def __iter__(self):
        for i, p in enumerate(self.paragraph):
            yield Paragraph(p, batch_id=i, version=self.version)

    def __len__(self):
        return self.num_qa

    def get_num_qa(self):
        return sum([p.batch_size for p in self])


class Paragraph(object):
    def __init__(self, p, batch_id=None, version="1.1", device="cuda"):
        self.context = p["context"]
        self.version = version
        self.expand_qas(p)
        self.batch_size = len(self.q)
        self.batch_id = batch_id

    def expand_qas(self, p):
        self.q = []
        self.a = []
        self.a_index = []
        self.id = []
        if self.version == "2.0":
            self.impossible = []
        for qa in p["qas"]:
            self.q.append(qa["question"])
            self.a.append(qa["answers"][0]["text"]) 
            self.a_index.append(qa["answers"][0]["answer_start"])
            self.id.append(qa["id"])
            if self.version == "2.0":
                self.impossible.append(qa["is_impossible"])

    def __len__(self):
        return self.batch_size



