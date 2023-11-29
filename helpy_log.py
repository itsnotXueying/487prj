import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def log_training(epoch, stats_at_epoch, info, log_dir):
    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(b = info["batch"],
                                                lr = info["lr"],
                                                p = info["p"],
                                                wd = info["wd"])
    model_path = os.path.join(log_dir,name)
    model_path = os.path.join(model_path,"loss_log.csv")

    if epoch == 0:
        if os.path.exists(model_path):
            os.remove(model_path)
        with open(model_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch","Train Loss","Val Loss","Test Loss"])
        
    with open(model_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, *stats_at_epoch])
        
