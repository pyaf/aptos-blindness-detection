import pdb
import os
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from models import Model, get_model
from utils import *
from image_utils import *


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--ckpt_path",
        dest="ckpt_path",
        help="Checkpoint to use",
    )
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="resnext101_32x4d",
    )
    return parser


class TestDataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.TTA = albumentations.Compose(
            [
                albumentations.Rotate(limit=180, p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.RandomScale(scale_limit=0.1),
                #albumentations.ShiftScaleRotate(
                #    shift_limit=0,  # no resizing
                #    scale_limit=0.1,
                #    rotate_limit=120,
                #    p=0.5,
                #    border_mode=cv2.BORDER_CONSTANT
                #),
                albumentations.OneOf(
                    [
                        albumentations.CLAHE(clip_limit=2),
                        albumentations.IAASharpen(),
                        albumentations.IAAEmboss(),
                        albumentations.RandomBrightnessContrast(),
                        albumentations.JpegCompression(),
                        albumentations.Blur(),
                        albumentations.GaussNoise(),
                    ],
                    p=0.5,
                ),
            ]
        )
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                albumentations.Resize(size, size),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, -1)
        image = np.repeat(image, 3, -1)
        images = [self.transform(image=image)["image"]]
        for _ in range(self.tta):  # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples



def get_predictions(model, testset, tta):
    """return all predictions on testset in a list"""
    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        if tta:
            for images in batch:  # images.shape [n, 3, 96, 96] where n is num of 1+tta
                preds = torch.sigmoid(model(images.to(device))) # [n, num_classes]
                preds = preds.mean(dim=0).detach().tolist()
                predictions.append(preds)
        else:
            preds = torch.sigmoid(model(batch[:, 0].to(device)))
            preds = preds.detach().tolist() #[1]
            predictions.extend(preds)

    return np.array(predictions)


def get_model_name_fold(ckpt_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/ckpt12.pth
    model_folder = ckpt_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2][1:]  # f0 -> 0
    return model_name, int(fold)


if __name__ == "__main__":
    '''
    use given ckpt to generate final predictions using the corresponding best thresholds.
    '''
    parser = get_parser()
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    predict_on = args.predict_on
    model_name, fold = get_model_name_fold(ckpt_path)

    if predict_on == "test":
        sample_submission_path = "../data/sample_submission.csv"
    else:
        sample_submission_path = "../data/train.csv"

    tta = 4 # number of augs in tta
    sub_path = ckpt_path.replace(".pth", f"{predict_on}.csv")
    root = f"../data/{predict_on}_png/"
    size = 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #mean = (0, 0, 0)
    #std = (1, 1, 1)
    use_cuda = True
    num_classes = 1
    num_workers = 8
    batch_size = 16
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(root, df, size, mean, std, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    model = get_model(model_name, num_classes, pretrained=None)
    model.to(device)
    model.eval()

    print(f"Using {ckpt_path}")
    print(f"Predicting on: {predict_on} set")
    print(f"Root: {root}")
    print(f"Using tta: {tta}\n")

    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    best_thresholds = state["best_thresholds"]
    print(f"Best thresholds: {best_thresholds}")
    preds = get_predictions(model, testset, tta)
    best_thresholds = 0.5
    pred_labels = predict(preds, best_thresholds)
    pdb.set_trace()
    df["label"] = pred_labels
    print(f"Saving predictions at {sub_path}")
    df.to_csv(sub_path, index=False)
    print("Predictions saved!")


'''
Footnotes

[1] a cuda variable can be converted to python list with .detach() (i.e., grad no longer required) then .tolist(), apart from that a cuda variable can be converted to numpy variable only by copying the tensor to host memory by .cpu() and then .numpy
'''