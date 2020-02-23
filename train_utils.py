import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import evaluation
from seqeval.metrics import f1_score


def train(train_data, val_data, model, lr=1e-3, patience=10, max_epoch=100,
          print_freq=1):
    no_improvement = 0
    best_val_f1 = 0
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, factor=0.1, patience=patience)
    for epoch in range(max_epoch):
        if no_improvement > patience:
            break
        running_loss = 0.0
        model.train() # turn on training mode
        for batch in train_data: 
            opt.zero_grad()
            preds = model(batch)
            try:
                loss = loss_func(preds, batch.label)
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
        if epoch % print_freq == 0:
            print('Epoch: {}, LR: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val f1 {:.3f}'.format(
                epoch, opt.param_groups[0]['lr'], epoch_loss, val_loss, val_f1))
    train_loss, train_f1 = calculate_score(train_data, best_model, loss_func)
    result = {"trained_model": best_model, 
              "train f1 score": train_f1, 
              "val f1 score": best_val_f1, 
              "train loss": train_loss, 
              "val loss": val_loss}
    return result


def calculate_score(val_data, model, loss_func):
    idx2label = {i:label for i, label in enumerate(np.load("./data/labels.np"))}
    model.eval() 
    val_loss = 0.0
    y_pred = []
    y_true = []
    for batch in val_data:
        preds = model(batch)
        loss = loss_func(preds, batch.label)
        val_loss += loss.item() * batch.batch_size
        labels = [idx2label[i.item()] for i in torch.argmax(preds, dim=1)]
        assert(len(labels) == len(batch.raw_label))
        print("y true", batch.raw_label)
        print("y pred", labels)
        y_pred.append(labels)
        y_true.append(batch.raw_label)
    val_loss /= len(val_data)
    f1 = f1_score(y_true, y_pred)
    return val_loss, f1


    
