import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import evaluation
from seqeval.metrics import f1_score


def train(train_data, val_data, model, lr=1e-3, patience=10, max_epoch=100,
          print_freq=1):
    t00 = time.time()
    no_improvement = 0
    best_val_f1 = 0
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, factor=0.1, patience=patience)
    sec_per_epoch = []
    for epoch in range(max_epoch):
        t0 = time.time() 
        if no_improvement > patience:
            break
        running_loss = 0.0
        model.train() # turn on training mode
        for batch in train_data:
            opt.zero_grad()
            preds = model(batch)
            label = batch.label.reshape(-1)
            preds = preds.reshape(label.shape[0], -1)
            try:
                loss = loss_func(preds, label)
            except:
                print(preds.shape)
                print(batch.label.shape)
                print(batch.raw_label)
                print(batch.raw_text)
                raise
            loss.backward()
            opt.step()
            running_loss += loss.item() * batch.batch_size 
        epoch_loss = running_loss / len(train_data)
        val_loss, val_f1 = calculate_score(val_data, model, loss_func) 
        
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
    train_loss, train_f1 = calculate_score(train_data, best_model, loss_func)
    result = {"trained_model": best_model, 
              "train f1 score": train_f1, 
              "val f1 score": best_val_f1, 
              "train loss": train_loss, 
              "val loss": val_loss,
              "epoch_time": sec_per_epoch,
              "total_time": time.time() - t00,
              }
    return result


def calculate_score(val_data, model, loss_func):
    idx2label = {i:label for i, label in enumerate(np.load("./data/labels.npy"))}
    model.eval() 
    val_loss = 0.0
    y_pred = []
    y_true = []
    for batch in val_data:
        preds = model(batch)
        loss = loss_func(preds.reshape(-1, preds.shape[2]), 
                         batch.label.reshape(-1))
        val_loss += loss.item() * batch.batch_size
        labels = [[idx2label[j.item()] for j in i] 
                  for i in torch.argmax(preds, dim=2)]
        y_pred.extend(labels)
        y_true.extend(batch.raw_label)
    val_loss /= len(val_data)
    f1 = f1_score(y_true, y_pred)
    return val_loss, f1

