import os
import pdb
import cv2
import time
import torch
import scipy
import logging
import traceback
import numpy as np
from datetime import datetime

# from config import HOME
from tensorboard_logger import log_value, log_images
from torchnet.meter import ConfusionMeter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix

plt.switch_backend("agg")


def logger_init(save_folder):
    mkdir(save_folder)
    logging.basicConfig(
        filename=os.path.join(save_folder, "log.txt"),
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    return logger


def to_multi_label(target, classes):
    '''[0, 0, 1, 0] to [1, 1, 1, 0]'''
    multi_label = np.zeros((len(target), classes))
    for i in range(len(target)):
        j = target[i] + 1
        multi_label[i][:j] = 1
    return np.array(multi_label)


def get_preds(arr, num_cls):
    ''' takes in thresholded predictions (num_samples, num_cls) and returns (num_samples,)
    [3], arr needs to be a numpy array, NOT torch tensor'''
    mask = arr == 0
    #pdb.set_trace()
    return np.clip(np.where(mask.any(1), mask.argmax(1), num_cls) - 1, 0, num_cls-1)


def compute_score_inv(threshold, predictions, targets, num_classes):
    predictions = predictions > threshold
    predictions = get_preds(predictions, num_classes)
    score = cohen_kappa_score(predictions, targets, weights='quadratic')
    return 1 - score


class Meter:
    def __init__(self, phase, epoch, save_folder):
        self.predictions = []
        self.targets = []
        self.phase = phase
        self.epoch = epoch
        self.save_folder = os.path.join(save_folder, "logs")
        self.num_classes = 5 # hard coded, yeah, I know
        self.best_threshold = 0.5

    def update(self, targets, outputs):
        # get multi-label to single label
        targets = torch.sum(targets, 1) - 1
        targets = targets.type(torch.LongTensor)
        #outputs = torch.sum((outputs > 0.5), 1) - 1

        #pdb.set_trace()
        self.targets.extend(targets.tolist())
        self.predictions.extend(outputs.tolist())
        #self.predictions.extend(torch.argmax(outputs, dim=1).tolist()) #[2]

    def get_best_threshold(self):
        '''Used in the val phase of iteration, see [4]'''
        self.predictions = np.array(self.predictions)
        self.targets = np.array(self.targets)
        simplex = scipy.optimize.minimize(
            compute_score_inv,
            0.5,
            args=(self.predictions, self.targets, self.num_classes),
            method='nelder-mead'
        )
        self.best_threshold = simplex['x'][0]
        print("Best threshold: %s" % self.best_threshold)
        return self.best_threshold

    def get_cm(self):
        threshold = self.best_threshold
        self.predictions = np.array(self.predictions) > threshold # [5]
        self.predictions = get_preds(self.predictions, self.num_classes)
        cm = ConfusionMatrix(self.targets, self.predictions)
        qwk = cohen_kappa_score(self.targets, self.predictions, weights="quadratic")
        return cm, qwk


def plot_ROC(roc, targets, predictions, phase, epoch, folder):
    roc_plot_folder = os.path.join(folder, "ROC_plots")
    mkdir(os.path.join(roc_plot_folder))
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_plot_name = "ROC_%s_%s_%0.4f" % (phase, epoch, roc)
    roc_plot_path = os.path.join(roc_plot_folder, roc_plot_name + ".jpg")
    fig = plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.legend(["diagonal-line", roc_plot_name])
    fig.savefig(roc_plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # see footnote [1]

    plot = cv2.imread(roc_plot_path)
    log_images(roc_plot_name, [plot], epoch)


def print_time(log, start, string):
    diff = time.time() - start
    log(string + ": %02d:%02d" % (diff // 60, diff % 60))


def adjust_lr(lr, optimizer):
    """ Update the lr of base model
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups[:-1]:
        param_group["lr"] = lr
    return optimizer


def iter_log(log, phase, epoch, iteration, epoch_size, loss, start):
    diff = time.time() - start
    log(
        "%s epoch: %d (%d/%d) loss: %.4f || %02d:%02d",
        phase,
        epoch,
        iteration,
        epoch_size,
        loss.item(),
        diff // 60,
        diff % 60,
    )


def epoch_log(log, phase, epoch, epoch_loss, meter, start):
    diff = time.time() - start
    cm, qwk = meter.get_cm()
    acc = cm.overall_stat["Overall ACC"]
    log("<===%s epoch: %d finished===>" % (phase, epoch))
    log(
        "%s %d |  loss: %0.4f | qwk: %0.4f | acc: %0.4f"
        % (phase, epoch, epoch_loss, qwk, acc)
    )
    log(cm.print_normalized_matrix())
    log("Time taken for %s phase: %02d:%02d \n", phase, diff // 60, diff % 60)
    log_value(phase + " loss", epoch_loss, epoch)
    log_value(phase + " acc", acc, epoch)
    log_value(phase + " qwk", qwk, epoch)
    obj_path = os.path.join(meter.save_folder, f"cm{phase}_{epoch}")
    cm.save_obj(obj_path, save_stat=True, save_vector=False)

    # log_value(phase + " roc", roc, epoch)
    # log_value(phase + " precision", precision, epoch)
    # log_value(phase + " tnr", tnr, epoch)
    # log_value(phase + " fpr", fpr, epoch)
    # log_value(phase + " fnr", fnr, epoch)
    # log_value(phase + " tpr", tpr, epoch)


def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def save_hyperparameters(trainer, remark):
    hp_file = os.path.join(trainer.save_folder, "parameters.txt")
    time_now = datetime.now()
    # pdb.set_trace()
    with open(hp_file, "a") as f:
        f.write(f"Time: {time_now}\n")
        f.write(f"model_name: {trainer.model_name}\n")
        f.write(f"images_folder: {trainer.images_folder}\n")
        f.write(f"resume: {trainer.resume}\n")
        f.write(f"folder: {trainer.folder}\n")
        f.write(f"fold: {trainer.fold}\n")
        f.write(f"total_folds: {trainer.total_folds}\n")
        f.write(f"size: {trainer.size}\n")
        f.write(f"top_lr: {trainer.top_lr}\n")
        f.write(f"base_lr: {trainer.base_lr}\n")
        f.write(f"num_workers: {trainer.num_workers}\n")
        f.write(f"batchsize: {trainer.batch_size}\n")
        f.write(f"momentum: {trainer.momentum}\n")
        f.write(f"mean: {trainer.mean}\n")
        f.write(f"std: {trainer.std}\n")
        f.write(f"start_epoch: {trainer.start_epoch}\n")
        f.write(f"batchsize: {trainer.batch_size}\n")
        f.write(f"remark: {remark}\n")


"""Footnotes:

[1]: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures

[2]: Used in cross-entropy loss, one-hot to single label

[3]: # argmax returns earliest/first index of the maximum value along the given axis
 get_preds ka ye hai ki agar kisi output me zero nahi / sare one hain to 5 nahi to jis index par pehli baar zero aya wahi lena hai, example:
[[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
-> [4, 1, 0, 0]
baki clip karna hai (0, 4) me, we can get -1 for cases with all zeros.

[4]: get_best_threshold is used in the validation phase, during each phase (train/val) outputs and targets are accumulated. At the end of train phase a threshold of 0.5 is used for
generating the final predictions and henceforth for the computation of different metrics.
Now for the validation phase, best_threshold function is used to compute the optimum threshold so that the qwk is minimum and that threshold is used to compute the metrics.

It can be argued ki why are we using 0.5 for train, then, well we used 0.5 for both train/val so far, so if we are computing this val set best threshold, then not only it can be used to best evaluate the model on val set, it can also be used during the test time prediction as it is being saved with each ckpt.pth

[5]: np.array because it's a list and gets converted to np.array in get_best_threshold function only which is called in val phase and not training phase
"""
