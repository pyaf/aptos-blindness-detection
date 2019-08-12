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
from utils import to_multi_label
from extras import *
from augmentations import get_transforms


class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, phase, cfg):
        """
        Args:
                fold: for k fold CV
                images_folder: the folder which contains the images
                df_path: data frame path, which contains image ids
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.phase = phase
        self.df = df
        self.num_samples = self.df.shape[0]
        self.fnames = self.df["id_code"].values
        self.labels = self.df["diagnosis"].values.astype("int64")
        self.num_classes = len(np.unique(self.labels))
        # self.labels = to_multi_label(self.labels, self.num_classes)  # [1]
        # self.labels = np.eye(self.num_classes)[self.labels]
        self.transform = get_transforms(phase, cfg)
        self.root = cfg['data_folder']

        '''
        self.images = []
        for fname in tqdm(self.fnames):
            path = os.path.join(self.images_folder, "bgcc300", fname + ".npy")
            image = np.load(path)
            self.images.append(image)
        '''

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = self.labels[idx]
        path = os.path.join(self.root, fname + ".npy")
        image = np.load(path)
        #image = self.images[idx]
        image = self.transform(image=image)["image"]
        return fname, image, label

    def __len__(self):
        #return 1000
        return len(self.df)


def get_sampler(df, cfg):
    if cfg['cw_sampling']:
        '''sampler using class weights (cw)'''
        class_weights = cfg['class_weights']
        print("weights", class_weights)
        dataset_weights = [class_weights[idx] for idx in df["diagnosis"]]
        datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    if cfg['he_sampling']:
        '''sampler using hard examples (he)'''
        print('Hard example sampling')
        dataset_weights = df["weight"].values
        datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    else:
        datasampler = None
    return datasampler

def resampled(df, cfg):
    ''' resample from df with replace=False'''
    def sample(obj):  # [5]
        return obj.sample(n=count_dict[obj.name], replace=False, random_state=69)

    count_dict = cfg['count_dict']
    sampled_df = df.groupby('diagnosis').apply(sample).reset_index(drop=True)

    return sampled_df


def provider(phase, cfg):
    fold = cfg['fold']
    total_folds = cfg['total_folds']
    data_folder = cfg['data_folder']
    df_path = cfg['df_path']
    class_weights = eval(cfg['class_weights'])
    batch_size = cfg['batch_size'][phase]
    num_workers = cfg['num_workers']
    num_samples = cfg['num_samples']

    df = pd.read_csv(df_path)
    HOME = os.path.abspath(os.path.dirname(__file__))
    df['weight'] = 1 # [10]
    if cfg['he_sampling']:
        hard_examples = pd.read_csv(cfg['hard_df']).index.tolist()
        #hard_examples = np.load('data/hard_examples1.npy')  # [9]
        df.at[hard_examples, 'weight'] = cfg['hard_ex_weight']

    if cfg['tc_dups']:
        bad_indices = np.load(os.path.join(HOME, cfg["bad_idx"]))
        dup_indices = np.load(os.path.join(HOME, cfg["dups_wsd"]))  # [3]
        duplicates = df.iloc[dup_indices]
        all_dups = np.array(list(bad_indices) + list(dup_indices))
        df = df.drop(df.index[all_dups])  # remove duplicates and split train/val

    #''' to be used only with old data training '''
    if cfg['sample']:
        df = resampled(df, cfg)
        print(f'sampled df shape: {df.shape}')
        print('data dist:\n',  df['diagnosis'].value_counts(normalize=True))

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["id_code"], df["diagnosis"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    if cfg['tc_dups']:
        train_df = train_df.append(duplicates, ignore_index=True)  # add all

    if cfg['messidor_in_train']:
        mes_df = pd.read_csv(cfg['mes_df'])
        mes_df['weight'] = 1
        train_df = train_df.append(mes_df, ignore_index=True)

    if 'folder' in cfg.keys():
        # save for analysis, later on
        train_df.to_csv(os.path.join(cfg['folder'], 'train.csv'), index=False)
        val_df.to_csv(os.path.join(cfg['folder'], 'val.csv'), index=False)

    if phase == "train":
        df = train_df.copy()
    elif phase == "val":
        df = val_df.copy()
    elif phase == "val_new":
        df = pd.read_csv('data/train.csv')
    #df = pd.read_csv(cfg['diff_path'])
    print(f"{phase}: {df.shape}")

    image_dataset = ImageDataset(df, phase, cfg)

    datasampler = None
    if phase == "train":
        datasampler = get_sampler(df, cfg)
    print('datasampler:', datasampler)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False if datasampler else True,
        sampler=datasampler,
    )  # shuffle and sampler are mutually exclusive args

    #print(f'len(dataloader): {len(dataloader)}')
    return dataloader


if __name__ == "__main__":
    ''' doesn't work, gotta set seeds at function level
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    '''


    import time
    start = time.time()
    phase = "train"
    args = get_parser()
    cfg = load_cfg(args)
    dataloader = provider(phase, cfg)
    ''' train val set sanctity
    #pdb.set_trace()
    tdf = dataloader.dataset.df
    phase = "val"
    dataloader = provider(phase, cfg)
    vdf = dataloader.dataset.df
    print(len([x for x in tdf.id_code.tolist() if x in vdf.id_code.tolist()]))
    exit()
    '''
    total_labels = []
    total_len = len(dataloader)
    from collections import defaultdict
    fnames_dict = defaultdict(int)
    for idx, batch in enumerate(dataloader):
        fnames, images, labels = batch
        for fname in fnames:
            fnames_dict[fname] += 1

        print("%d/%d" % (idx, total_len), images.shape, labels.shape)
        total_labels.extend(labels.tolist())
        #pdb.set_trace()
    print(np.unique(total_labels, return_counts=True))
    diff = time.time() - start
    print('Time taken: %02d:%02d' % (diff//60, diff % 60))

    print(np.unique(list(fnames_dict.values()), return_counts=True))
    #pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2] .value_counts() returns in descending order of counts (not sorted by class numbers :)
[3]: bad_indices are those which have conflicting diagnosises, duplicates are those which have same duplicates, we shouldn't let them split in train and val set, gotta maintain the sanctity of val set
[4]: used when the dataframe include external data and we want to sample limited number of those
[5]: as replace=False,  total samples can be a finite number so that those many number of classes exist in the dataset, and as the count_dist is approx, not normalized to 1, 7800 is optimum, totaling to ~8100 samples

[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
[7]: indices of hard examples, evaluated using 0.81 scoring model.
[8]: messidor df append will throw err when doing hard ex sampling.
"""
