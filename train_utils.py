import copy
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import datetime
import os

import evaluation
from seqeval.metrics import f1_score


def train(train_data, val_data, model, model_name="distilbert", 
          lr=1e-5, patience=5, scheduler_patience=10, max_epoch=100,
          print_freq=1, print_batch=False, save_checkpt=True):
    t00 = time.time()
    filename = "dummy"
    no_improvement = 0
    best_val_f1 = 0
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, factor=0.1, patience=scheduler_patience)
    sec_per_epoch = []
    for epoch in range(max_epoch):
        t0 = time.time() 
        if no_improvement > patience:
            break
        running_loss = 0.0
        model.train() # turn on training mode
        for batch in train_data:
            if batch.batch_id % 1000 == 0:
                batch_t0 = time.time()
                print("batch", batch.batch_id, end=",")
                torch.cuda.empty_cache()
                if save_checkpt:
                    if os.path.exists(filename):
                        os.remove(filaname)
                    timestamp = datetime.datetime.today().strftime("%m%d%H%M")
                    filename = "{}/data/model_checkpoints/{}_{}.mdl".format(
                        os.getcwd(), model_name, timestamp)
                    torch.save(model, filename)
            opt.zero_grad()
            pred_answers, loss, _ = model(batch)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if print_batch and batch.batch_id % 1000 == 0:
                batch_time = time.time() - batch_t0
                epoch_time = batch_time/batch.batch_size * train_data.num_qa
                print("estimated epoch time: {}s".format(epoch_time))
        epoch_loss = running_loss/len(train_data)
        val_loss, val_f1 = calculate_score(val_data, model) 
        
        if val_f1 > best_val_f1:
            no_improvement = 0
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model)
        else:
            no_improvement += 1
        scheduler.step(val_loss)
        t_delta = time.time() - t0
        if epoch % print_freq == 0:
            print('Epoch: {}, LR: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val f1 {:.3f}, epoch time: {:.1f}s'.format(
                epoch, opt.param_groups[0]['lr'], epoch_loss, val_loss, val_f1, t_delta))
        sec_per_epoch.append(t_delta)
    train_loss, train_f1 = calculate_score(train_data, best_model)
    result = {"trained_model": best_model, 
              "train f1 score": train_f1, 
              "val f1 score": best_val_f1, 
              "train loss": train_loss, 
              "val loss": val_loss,
              "epoch_time": sec_per_epoch,
              "total_time": time.time() - t00,
              }
    return result


def calculate_score(data, model):
    model.eval() 
    answers = {}
    running_loss = 0
    for batch in data:
        pred_answers, loss, _ = model(batch)
        running_loss += loss.item()
        answers.update(pred_answers)    
    final_loss = running_loss / len(data)
    score = evaluation.get_score(data.data, answers)    
    return final_loss, score["f1"]


def inference(data, model):
    t0 = time.time()
    model.eval() 
    answers = {}
    for batch in data:
        answers.update(model(batch))
    score = evaluation.get_score(data.data, answers)    
    t_delta = time.time() - t0
    print("inference time ({0} paragraphs): {1:.1f} sec".format(len(data), t_delta))
    print("exact match: {}, f1 score: {}".format(score["exact"], score["f1"]))
    return score

