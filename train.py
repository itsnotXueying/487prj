import torch
import torchvision
import os
import matplotlib as plt
import numpy as np
import pandas as pd
import random
from dataset import *
from helpy_train import *
import helpy_log
import pdb
from torch.utils.data import Dataset, DataLoader
import gc
import warnings
import sys
import time
import argparse
from torchvision.models.densenet import densenet121, DenseNet121_Weights
torch.manual_seed(88)
np.random.seed(88)
random.seed(88)

warnings.filterwarnings("ignore")

def trial(batch_size_in, learning_rate_in, momentum_in, weight_decay_in, gpu):
    print(f'save_folder:{save_folder}')
    
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    
    tr_loader,va_loader,te_loader  = get_train_val_test_loaders(batch_size_in)
    
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 80)
    model.to(device)

    
    start_epoch = 0
    stats = []
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate_in,momentum=momentum_in,weight_decay=weight_decay_in)
    scaler = torch.cuda.amp.GradScaler()
    axes,fig = helpy_log.make_training_plot(batch_size_in, learning_rate_in, momentum_in, weight_decay_in)

    saved_path = os.path.join(save_folder,
                              f"b{batch_size_in}_lr{learning_rate_in}_p{momentum_in}_wd{weight_decay_in}")
    info = {"batch":batch_size_in, "lr":learning_rate_in,"p": momentum_in, "wd": weight_decay_in}
    
    if not os.path.exists(saved_path):
        os.makedirs(saved_path, exist_ok=True)
    
    
    print("inital eval")
    evaluate_epoch(axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats,
                   device,info, save_folder)


    global_min_loss = stats[-1][-2]
    
    patience = 5
    curr_count_to_patience = 0
    
    # Loop over the entire dataset multiple times
    epoch = start_epoch
    print(f"Entering train loop for lr:{learning_rate_in} p:{momentum_in} wd:{weight_decay_in}")
    while curr_count_to_patience < patience:
        print(f"starting epoch {epoch}")
        
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer, device, scaler)

        # # Evaluate model
        evaluate_epoch(axes, tr_loader, va_loader, te_loader,model, criterion, epoch + 1, stats,
                       device, info, save_folder)

        # # Save model parameters
        save_checkpoint(model, epoch + 1, save_folder, stats, info)

        if epoch > 8:
            curr_count_to_patience, global_min_loss = early_stopping(stats, curr_count_to_patience, global_min_loss)
        epoch += 1
    
    print(f"Finished Training after {epoch} epochs")
    helpy_log.save_cnn_training_plot(batch_size_in, learning_rate_in, momentum_in, weight_decay_in,False, save_folder)
    helpy_log.hold_training_plot()


parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-b", "--batch_size")
parser.add_argument("-lr", "--learning_rate")
parser.add_argument("-p", "--momentum")
parser.add_argument("-wd", "--weight_decay")
parser.add_argument("-g", "--gpu")
parser.add_argument("-s", "--save_folder")
# parser.add_argument("-a", "--all_layers")

args = parser.parse_args()


batch_size = int(args.batch_size)
learning_rate = float(args.learning_rate)
momentum = float(args.momentum)
weight_decay = float(args.weight_decay)
gpu = str(args.gpu)
save_folder = args.save_folder



print("starting training")
start_time = time.time()
trial(batch_size, learning_rate, momentum, weight_decay, gpu)
end_time = time.time()
name = f"b{batch_size}_lr{learning_rate}_p{momentum}_wd{weight_decay}"
time_dir = os.path.join(save_folder, name)
time_dir = os.path.join(time_dir,"time.txt")
with open(time_dir,"w") as file:
    file.write(str(end_time - start_time))

print("done!")

