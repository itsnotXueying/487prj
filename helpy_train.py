import numpy as np
import itertools
import os
import sys
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import helpy_log

def early_stopping(stats, curr_count_to_patience, global_min_loss):
    if stats[-1][-2] >= global_min_loss:
        curr_count_to_patience += 1
    else:
        global_min_loss = stats[-1][-2]
        curr_count_to_patience = 0
    return curr_count_to_patience, global_min_loss


def evaluate_epoch(tr_loader,val_loader,te_loader,model,criterion,epoch,
                   stats,device,info,save_folder,reg,include_test=True,
                   update_plot=True,multiclass=False):
    def _get_metrics(loader,is_train):
        y_true, y_score = [], []
        correct, total = 0, 0
        running_loss = []
        model.eval()
        max_iterations = 50
        iteration = 0
        for X, y in loader:
            if is_train and iteration > max_iterations:
                break
            with torch.no_grad():
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                y = y.float()
                y_true.append(y)
                y_score.append(output.data)
                running_loss.append(criterion(output, y).cpu())
            iteration += 1
        y_true = torch.cat(y_true)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        measure = 0
        if not reg:
            measure = metrics.roc_auc_score(y_true.cpu(), y_score.cpu(),average=None)
        return np.round(measure, decimals=3), round(loss,3)

    train_aurocs, train_loss = _get_metrics(tr_loader,True)
    val_aurocs, val_loss  = _get_metrics(val_loader,False)
    te_aurocs, te_loss  = _get_metrics(te_loader,False)

    print(f'Epoch:{epoch}')
    print(f'train loss {train_loss}')
    print(f'val loss {val_loss}')
    print(f'test loss {te_loss}\n')

    if not reg:
        helpy_log.log_aurocs(epoch,train_aurocs,info,save_folder,'train')
        helpy_log.log_aurocs(epoch,val_aurocs,info,save_folder,'val')
        helpy_log.log_aurocs(epoch,te_aurocs,info,save_folder,'test')

    stats_at_epoch = [train_loss, val_loss, te_loss]
    stats.append(stats_at_epoch)
    
    helpy_log.log_training(epoch, stats_at_epoch, info, save_folder)
    

def train_epoch(data_loader, model, criterion, optimizer, device):
    model.train()              
    for i, (X, y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device)
        y = y.float()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()



