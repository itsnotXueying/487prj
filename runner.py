import torch
import os
import matplotlib as plt
import numpy as np
import pandas as pd
import sys
import random
from tqdm.auto import tqdm
from helpy_train import *
import helpy_log
from torch.utils.data import Dataset, DataLoader
import gensim.downloader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from dataset_old import *
from mlp import *
# from mlp_bucket import *
# from dataset_bucketing import *
#from dataset_ingre_bucket import *


punks = string.punctuation
punks = punks+ '``'+ "''"
stopword_list = stopwords.words('english')

torch.manual_seed(88)
np.random.seed(88)
random.seed(88)



def trial(batch_size_in, learning_rate_in, momentum_in, weight_decay_in, save_folder, reg):
    print(f'save_folder:{save_folder}')
    
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    
    tr_loader,va_loader,te_loader  = get_train_val_test_loaders(batch_size_in)
    
    model = MLP()
    model.to(device)

    
    start_epoch = 0
    stats = []
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate_in, weight_decay=weight_decay_in)

    #axes,fig = helpy_log.make_training_plot(batch_size_in, learning_rate_in, momentum_in, weight_decay_in)

    saved_path = os.path.join(save_folder,
                              f"b{batch_size_in}_lr{learning_rate_in}_p{momentum_in}_wd{weight_decay_in}")
    info = {"batch":batch_size_in, "lr":learning_rate_in,"p": momentum_in, "wd": weight_decay_in}
    
    if not os.path.exists(saved_path):
        os.makedirs(saved_path, exist_ok=True)
    
    
    print("inital eval")
    evaluate_epoch(tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats,
                   device,info, save_folder,reg)


    global_min_loss = stats[-1][-2]
    
    patience = 5
    curr_count_to_patience = 0
    
    # Loop over the entire dataset multiple times
    epoch = start_epoch
    print(f"Entering train loop for lr:{learning_rate_in} p:{momentum_in} wd:{weight_decay_in}")
    while curr_count_to_patience < patience:
        print(f"starting epoch {epoch}")
        
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer, device)

        # Evaluate model
        evaluate_epoch(tr_loader, va_loader, te_loader,model, criterion, epoch + 1, stats,
                       device, info, save_folder,reg)

        # Save model parameters
        save_checkpoint(model, epoch + 1, save_folder, stats, info)

        if epoch > 2:
            curr_count_to_patience, global_min_loss = early_stopping(stats, curr_count_to_patience, global_min_loss)
        epoch += 1
    print(f"Finished Training after {epoch} epochs")

print("starting")
for lr in tqdm([1e-1,1e-2,1e-3]):
    for p in [0.8,0.9]:
        for wd in [1e-1,1e-2,1e-3]:
            if lr == wd:
                continue
            trial(64, lr, p, wd,'results_mlp_ingre_regression',reg=True)
print("DONE!")