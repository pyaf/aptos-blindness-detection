import os
import cv2
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision.datasets.folder import pil_loader
from sklearn.model_selection import train_test_split, StratifiedKFold

# from utils import *
import albumentations
from albumentations import torch as AT
from image_utils import *


class DRDataset(Dataset):
    def __init__(self, df, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)

        self.fnames = self.df["id_code"].tolist()
        self.labels = self.df["label"].tolist()  # .astype(np.int32) # [12]
        print(np.unique(self.labels, return_counts=True))

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        # image_path = os.path.join(self.root, "npy_train_256",  image_id + '.npy')
        # img = np.load(image_path)
        # img = np.repeat(img, 3, axis=-1)
        # print(img.shape)
        path = os.path.join(self.root, fname)
        # print(path)
        image = id_to_image(
            path,
            resize=True,
            size=self.size,
            augmentation=False,
            subtract_median=True,
            clahe_green=False,
        )
        # image = self.images[idx]

        augmented = self.transforms(image=image)  # , mask=mask)
        img = augmented["image"]  # / 255.0

        target = {}
        target["labels"] = self.labels[idx]
        target["image_id"] = fname
        # pdb.set_trace()
        return img, target

    def __len__(self):
        # return 20
        return len(self.fnames)


def get_transforms(phase, size, mean, std):
    list_transforms = [
        # albumentations.Resize(size, size) # now doing this in __getitem__()
    ]
    if phase == "train":
        list_transforms.extend(
            [
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=180,
                    p=0.5,
                    # border_mode=cv2.BORDER_CONSTANT
                ),
                albumentations.RandomBrightnessContrast(p=1),
            ]
        )
    list_transforms.extend(
        [
            albumentations.Normalize(mean=mean, std=std, p=1),
            # albumentations.Resize(size, size),
            AT.ToTensor(normalize=None),  # [6]
        ]
    )
    return albumentations.Compose(list_transforms)


def get_sampler(df, class_weights=None):
    if class_weights is None:
        labels, label_counts = np.unique(
            df["diagnosis"].values, return_counts=True
        )  # [2]
        # class_weights = max(label_counts) / label_counts # higher count, lower weight
        # class_weights = class_weights / sum(class_weights)
        class_weights = [1, 1, 1, 1, 1]
    print("weights", class_weights)
    dataset_weights = [class_weights[idx] for idx in df["diagnosis"]]
    datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    return datasampler


def provider(
    fold,
    total_folds,
    images_folder,
    df_path,
    phase,
    size,
    mean,
    std,
    class_weights=None,
    batch_size=8,
    num_workers=4,
    num_samples=4000,
):
    df = pd.read_csv(df_path)
    df["label"] = df.diagnosis.apply(lambda x: 1 if x == 4 else 0)
    df_with_4 = df[df["diagnosis"] == 1]
    df_without_4 = df[df["diagnosis"] != 1]
    df_without_4_sampled = df_without_4.sample(1000)
    df = pd.concat([df_with_4, df_without_4_sampled])

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["id_code"], df["label"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    image_dataset = DRDataset(df, images_folder, size, mean, std, phase)
    # datasampler = get_sampler(df, [1, 1])
    datasampler = None
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False if datasampler else True,
        sampler=datasampler,
    )  # shuffle and sampler are mutually exclusive args

    # print(f'len(dataloader): {len(dataloader)}')
    return dataloader


if __name__ == "__main__":
    import time

    start = time.time()
    phase = "train"
    # phase = "val"
    num_workers = 12
    fold = 0
    total_folds = 5
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)

    size = 300

    root = os.path.dirname(__file__)  # data folder
    data_folder = "../data/all_images/"
    df_path = "../data/train.csv"
    num_samples = None  # 5000
    class_weights = True  # [1, 1, 1, 1, 1]
    batch_size = 16
    # images_folder = os.path.join(root, data_folder, "train_png/")  #

    dataloader = provider(
        fold,
        total_folds,
        data_folder,
        df_path,
        phase,
        size,
        mean,
        std,
        class_weights=class_weights,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )
    total_labels = []
    total_len = len(dataloader)
    from collections import defaultdict

    fnames_dict = defaultdict(int)
    for idx, batch in enumerate(dataloader):
        images, targets = batch
        labels = targets["labels"]
        # pdb.set_trace()
        for fname in targets["image_id"]:
            fnames_dict[fname] += 1

        print("%d/%d" % (idx, total_len), images.shape, labels.shape)
        total_labels.extend(labels.tolist())
    # pdb.set_trace()

    print("Unique label count:", np.unique(total_labels, return_counts=True))
    diff = time.time() - start
    print("Time taken: %02d:%02d" % (diff // 60, diff % 60))
    print(
        "fnames unique count:",
        np.unique(list(fnames_dict.values()), return_counts=True),
    )
    pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2] .value_counts() returns in descending order of counts (not sorted by class numbers :)
[3]: bad_indices are those which have conflicting diagnosises, duplicates are those which have same duplicates, we shouldn't let them split in train and val set, gotta maintain the sanctity of val set
[4]: used when the dataframe include external data and we want to sample limited number of those
[5]: as replace=False,  total samples can be a finite number so that those many number of classes exist in the dataset, and as the count_dist is approx, not normalized to 1, 7800 is optimum, totaling to ~8100 samples

[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
"""
