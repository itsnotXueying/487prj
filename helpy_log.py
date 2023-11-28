import os
import numpy as np
import matplotlib.pyplot as plt
import csv


def denormalize_image(image):
    """ Rescale the image's color space from (min, max) to (0, 1) """
    ptp = np.max(image, axis=(0, 1)) - np.min(image, axis=(0, 1))
    return (image - np.min(image, axis=(0, 1))) / ptp


def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()


def log_training(epoch, stats_at_epoch, info, log_dir):
    """Print the train, validation, test accuracy/loss/auroc.

    Each epoch in `stats` should have order
        [val_acc, val_loss, val_auc, train_acc, ...]
    Test accuracy is optional and will only be logged if stats is length 9.
    """
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
        
    

def log_aurocs(epoch,aurocs,info,log_dir,train_val):
    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(b = info["batch"],
                                                lr = info["lr"],
                                                p = info["p"],
                                                wd = info["wd"])
    model_path = os.path.join(log_dir,name)
    model_path = os.path.join(model_path,f"{train_val}_aurocs.csv")

    if epoch == 0:
        if os.path.exists(model_path):
            os.remove(model_path)
        with open(model_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch",'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'])
        
    with open(model_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, *aurocs])
    


def make_training_plot(batch_size_in, learning_rate_in, momentum_in, weight_decay_in):
    """Set up an interactive matplotlib graph to log metrics during training."""
    plt.ion()
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    name = "b:{b} lr:{lr} p:{p}_wd_{wd}".format(b = batch_size_in,
                                                         lr = learning_rate_in,
                                                         p = momentum_in,
                                                         wd = weight_decay_in)
    plt.suptitle(name)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")

    return axes, fig


def update_training_plot(axes, epoch, stats):
    """Update the training plot with a new data point for loss and accuracy."""
    splits = ["Train", "Validation"]
    metrics = ["Loss"]
    colors = ["r", "b"]
    for i, metric in enumerate(metrics):
        for j, split in enumerate(splits):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            axes.plot(
                range(epoch - len(stats) + 1, epoch + 1),
                [stat[idx] for stat in stats],
                linestyle="--",
                marker="o",
                color=colors[j],
            )
        axes.legend(splits[: int(len(stats[-1]) / len(metrics))])
    plt.pause(0.00001)


def save_cnn_training_plot(batch_size_in, learning_rate_in, momentum_in, weight_decay_in, only_classifier, save_folder):
    """Save the training plot to a file."""
    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(lr = learning_rate_in, p = momentum_in, b = batch_size_in, wd = weight_decay_in)

    save_path = os.path.join(save_folder,name)
    save_path = os.path.join(save_path, "plot.png")
    plt.savefig(save_path, dpi=200)


def save_tl_training_plot(num_layers):
    """Save the transfer learning training plot to a file."""
    if num_layers == 0:
        plt.savefig("TL_0_layers.png", dpi=200)
    elif num_layers == 1:
        plt.savefig("TL_1_layers.png", dpi=200)
    elif num_layers == 2:
        plt.savefig("TL_2_layers.png", dpi=200)
    elif num_layers == 3:
        plt.savefig("TL_3_layers.png", dpi=200)


def save_source_training_plot():
    """Save the source learning training plot to a file."""
    plt.savefig("source_training_plot.png", dpi=200)

def save_challenge_training_plot():
    """Save the challenge learning training plot to a file."""
    plt.savefig("challenge_training_plot.png", dpi=200)
