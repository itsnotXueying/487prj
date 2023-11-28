import numpy as np
import itertools
import os
import sys
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import helpy_log


def count_parameters(model):
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir, stats, info):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats}

    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(b = info["batch"],
                                                 lr = info["lr"],
                                                 p = info["p"],
                                                 wd = info["wd"])

    checkpoint_dir = os.path.join(checkpoint_dir, name)
    checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    

    filename = os.path.join(checkpoint_dir, f"epoch={epoch}.checkpoint.pth.tar")
    torch.save(state, filename)


def check_for_augmented_data(data_dir):
    """Ask to use augmented data if `augmented_dogs.csv` exists in the data directory."""
    if "augmented_dogs.csv" in os.listdir(data_dir):
        print("Augmented data found, would you like to use it? y/n")
        print(">> ", end="")
        rep = str(input())
        return rep == "y"
    return False


def restore_checkpoint(model, checkpoint_dir, cuda=True, force=False, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print(
            "Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(
            filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(
        checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def early_stopping(stats, curr_count_to_patience, global_min_loss):
    if stats[-1][-2] >= global_min_loss:
        curr_count_to_patience += 1
    else:
        global_min_loss = stats[-1][-2]
        curr_count_to_patience = 0
    return curr_count_to_patience, global_min_loss


def evaluate_epoch(axes,tr_loader,val_loader,te_loader,model,criterion,epoch,
                   stats,device,info,save_folder,include_test=True,
                   update_plot=True,multiclass=False,):
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

        return round(loss,5)

    train_aurocs, train_loss = _get_metrics(tr_loader,True)
    val_aurocs, val_loss  = _get_metrics(val_loader,False)
    te_aurocs, te_loss  = _get_metrics(te_loader,False)

    print(f'Epoch:{epoch}')
    print(f'train aurocs {train_aurocs[:3]}')
    print(f'val aurocs {val_aurocs[:3]}')
    print(f'test aurocs {te_aurocs[:3]}')

    helpy_log.log_aurocs(epoch,train_aurocs,info,save_folder,'train')
    helpy_log.log_aurocs(epoch,val_aurocs,info,save_folder,'val')
    helpy_log.log_aurocs(epoch,te_aurocs,info,save_folder,'test')

    
    stats_at_epoch = [train_loss, val_loss, te_loss]
    stats.append(stats_at_epoch)
    
    helpy_log.log_training(epoch, stats_at_epoch, info, save_folder)
    if update_plot:
        helpy_log.update_training_plot(axes, epoch, stats)
    

def train_epoch(data_loader, model, criterion, optimizer, device):
    model.train()              
    for i, (X, y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()



