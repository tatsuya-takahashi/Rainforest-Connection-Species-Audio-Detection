# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Train Based on clipping

# %%
import os
PROJECT = "RFCX"
EXP_NUM = "32"
EXP_TITLE = "resnest50"
EXP_NAME = "exp_" + EXP_NUM + "_" + EXP_TITLE
IS_WRITRE_LOG = True
os.environ['WANDB_NOTEBOOK_NAME'] = 'train_clip'
print('expname:' + EXP_NAME)

# %% [markdown]
# ## Library

# %%
# library import
import numpy as np
import pandas as pd
# import os
# import tqdm
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import random
import time
import math
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import librosa
import wandb
from time import sleep
from torch.nn import functional as F
from torch.optim import Adam, AdamW
import torch_optimizer as toptim
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torch.utils.data as torchdata
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from contextlib import contextmanager
from typing import Optional
from numpy.random import beta
from pathlib import Path
from fastprogress.fastprogress import master_bar, progress_bar
from torchviz import make_dot
from conformer import ConformerConvModule
from conformer import ConformerBlock
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
from torchviz import make_dot
from torchsummary import summary
from torchlibrosa.augmentation import SpecAugmentation
import librosa.display
from resnest.torch import resnest50
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage import exposure, util
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# from IPython.core.display import display, HTML
# display(HTML("<style>.scroll_box { height:90em  !important; }</style>"))

# %% [markdown]
# ## Configuration

# %%
# use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# %%
# load weight
# model_efn = EfficientNet.from_pretrained('efficientnet-b7')
# model_efn = EfficientNet.from_pretrained('efficientnet-b7')
# model_efn = EfficientNet.from_pretrained('efficientnet-b4')
# model_efn.to(device); # calculate on cpu


# %%
# 5get length
class params:
    sr = 48000
    n_mels = 320
    fmin = 40
    fmax = sr // 2
    fft = 2048
    hop = 512
    clip_frame = 10 * 48000
    augnum = 100

def wav2mel(wavnp):
    melspec = librosa.feature.melspectrogram(
        wavnp, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax, n_fft=params.fft, hop_length=params.hop, 
    )
    melspec = librosa.power_to_db(melspec).astype(np.float32)

    # # normalize
    # melspec = melspec - np.min(melspec)
    # melspec = melspec / np.max(melspec)

    eps=1e-6 # avoid  divided by 0
    mean = melspec.mean()
    std = melspec.std()
    spec_norm = (melspec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    spec_scaled = np.asarray(spec_scaled)

    return spec_scaled

wavnp = np.load(Path('../input//rfcx-species-audio-detection/train_mel/0.npy'))
print(wavnp.shape)
sample = wav2mel(wavnp[0: 10 * params.sr]) # 10s clipping

# sample data
# sample = torch.load(Path("e:/002_datasets/000_RFCX/train_mel_clip_aug/0_0.pt"))
# sample = torch.from_numpy(np.load(Path("../input/rfcx-species-audio-detection/train_mel/0.npy")))
# sample = torch.load(Path("../input/rfcx-species-audio-detection/train_mel_clip_aug/0_0.pt"))

# channel, seq, dim
print(sample.shape)
# print(sample[np.newaxis, :, :].shape)
clip_len = int(sample.shape[1])
clip_dim = int(sample.shape[0])
print(clip_len, clip_dim)


# %%
# # expeliment
# clip = sample.T
# print("clip", clip.shape)

# # stacking
# img = torch.from_numpy(np.array([
#         [clip],[clip],[clip]
#     ])).float().transpose(0, 1)
# print("img", img.shape)

# # encoding
# enc = model_efn.extract_features(img.to(device))
# print("enc", enc.shape)

# enc = enc.detach().cpu()

# # save
# ch = enc.shape[1]
# enc_len = enc.shape[2]
# enc_dim = enc.shape[3]
# print('ch, enc_len, enc_dim', ch, enc_len, enc_dim)

# del enc

ch = 1792
enc_len = 30
enc_dim = 10


# %%
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)


# %%
def get_model():
    resnet_model = resnest50(pretrained=True)
    # resnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
    # resnet_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, config.NUM_BIRDS)
    # resnet_model = resnet_model.to(device)
    return resnet_model


# %%
class dict2(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self 


# %%
1e-3


# %%
config = dict2({
    "fft":                2048,
    "hop":                512,
    "sr":                 48000,
    "mel":                320,
    "SEED":               42,
    # "INPUT":              Path("../input/rfcx-species-audio-detection/train"),
    # "TRAIN_AUDIO_ROOT":   Path("../input/rfcx-species-audio-detection/train_mel_clip_aug/"),
    # "TEST_AUDIO_ROOT":    Path"../input/rfcx-species-audio-detection/train_mel_clip_aug/0_0.pt"),
    # "TRAIN_TP":           Path("../input/rfcx-species-audio-detection/train_tp.csv"),
    # "TRAIN_TP_MEL":       Path("../input/rfcx-species-audio-detection/train_tp_mel.csv"),
    # "SUB":                Path("../input/rfcx-species-audio-detection/sample_submission.csv"),
    "TEST_AUDIO_FLAC":    Path("../input/rfcx-species-audio-detection/test"),
    "TRAIN_AUDIO_ROOT":   Path("e:/002_datasets/000_RFCX/train_mel_clip_aug/"),
    "TEST_AUDIO_ROOT":    Path("../input/rfcx-species-audio-detection/test_mel"),
    "VALID_AUDIO_ROOT":   Path("e:/002_datasets/000_RFCX/valid_mel_clip/"),
    "TRAIN_TP":           Path("../input/rfcx-species-audio-detection/train_tp.csv"),
    "TRAIN_TP_CSV":       Path("../input/rfcx-species-audio-detection/train_tp_mel.csv"),
    "VALID_CSV":          Path("../input/rfcx-species-audio-detection/valid.csv"),
    "TEST_CSV":           Path("../input/rfcx-species-audio-detection/test.csv"),
    "SUB":                Path("../input/rfcx-species-audio-detection/sample_submission.csv"),
    # "DIM":                sample.shape[0],
    # "SEQ_LEN":            int(sample.shape[1] * 0.8),
    # "DIM":                dim,
    # "ENC_LEN":            seq_len,
    "MIX_LABEL":          0.0,
    "CLIP_LEN":           clip_len,
    "CLIP_DIM":           clip_dim,
    "ENC_CH":             ch,
    "ENC_LEN":            enc_len,
    "ENC_DIM":            enc_dim,
    "KERNEL_SIZE":        3,
    "KERNEL_STRIDE":      1,
    "KERNEL_SIZE_SEQ":    3,
    "POOL_SIZE":          2,
    "POOL_STRIDE":        2,
    "NUM_BIRDS":          24,
    "N_FOLDS":            5,
    "BATCH_NUM":          22,
    "VALID_BATCH_NUM":    22,
    "EPOCH_NUM":          30,
    "DROPOUT":            0.35,
    "lr": 1e-3,
    "momentum": 0.9,
    "gamma": 0.7,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0,
    "t_max":              10,
    "TEST_SIZE":          0.2,
    "MIXUP":              0.0,
    "MIXUP_PROB":         -1.0,
    "SPEC_PROB":          -1,
    "spec_time_w":        0,
    "spec_time_stripes":  0,
    "spec_freq_w":        0,
    "spec_freq_stripes":  0,
})


# %%
# seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
set_seed(config.SEED)


# %%
spec_augmenter = SpecAugmentation(time_drop_width=config.spec_time_w, time_stripes_num=config.spec_time_stripes, 
            freq_drop_width=config.spec_freq_w, freq_stripes_num=config.spec_freq_stripes)


# %%
# print(img.shape)
# print(spec_augmenter(img).shape)


# %%
# # stacking
# img = torch.from_numpy(np.array([
#         [clip],[clip],[clip]
#     ])).float().transpose(0, 1)

# print(img.shape)


# s_img = spec_augmenter(img)

# fig, ax = plt.subplots(figsize=(15, 5))
# figure1ch = librosa.display.specshow(
#     s_img.numpy()[0][0].T, 
#     sr=48000,
#     x_axis='time', 
#     y_axis='linear', 
#     ax=ax)
# fig, ax = plt.subplots(figsize=(15, 5))
# figure2ch = librosa.display.specshow(
#     s_img.numpy()[0][1].T, 
#     sr=48000,
#     x_axis='time', 
#     y_axis='linear', 
#     ax=ax)
# fig, ax = plt.subplots(figsize=(15, 5))
# figure3ch = librosa.display.specshow(
#     s_img.numpy()[0][2].T, 
#     sr=48000,
#     x_axis='time', 
#     y_axis='linear', 
#     ax=ax)

# %% [markdown]
# ## Augment

# %%
# Data load
df_train_tp = pd.read_csv(config.TRAIN_TP_CSV)

duplicate_recids = df_train_tp.groupby("recording_id", as_index=False).count()[df_train_tp.groupby("recording_id", as_index=False).count().id > 1].recording_id.values

specids = {}
for index, row in df_train_tp.iterrows():
    for duprecid in duplicate_recids:
        if duprecid == row["recording_id"]:
            if duprecid not in specids:
                specids[duprecid] = []
            specids[duprecid].append(row["species_id"])

spec2spec = {}
for i in range(24):
    spec2spec[i] = []
       
for s in range(24):
    for specs in specids.values():
        if s in specs:
            for spec in specs:
                if s != spec:
                    spec2spec[s].append(spec)
display(spec2spec)

def s2s(specid):
    if len(spec2spec[specid]) > 0:
        return np.random.choice(spec2spec[specid])
    else:
        return specid

# print(s2s(3))
# print(s2s(10))

spec2id = {}
for index, row in df_train_tp.iterrows():
    if row["species_id"] not in spec2id:
        spec2id[row["species_id"]] = []
    spec2id[row["species_id"]].append(row["id"])

def s2id(specid):
    return np.random.choice(spec2id[specid])

# print(spec2id[0])
# print(s2id(0))


# %%
# # mixup
# def mixup_data(x, y, alpha=1.0, use_cuda=True):

#     '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0.:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1.
#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)
#     # lam = max(lam, 1 - lam)
#     mixed_x = lam * x + (1 - lam) * x[index,:]
#     # mixed_y = lam * y + (1 - lam) * y[index]
#     y_a, y_b = y, y[index]
#     # return mixed_x, mixed_y
#     return mixed_x, y_a, y_b, lam

# # def mixup_criterion(y_a, y_b, lam):
# #     return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# %%
# mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]


    specids = torch.max(y, dim=1).detach.cpu().numpy()
    for specid in specids:
        


    # lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    # mixed_y = lam * y + (1 - lam) * y[index]
    y_a, y_b = y, y[index]
    # return mixed_x, mixed_y
    return mixed_x, y_a, y_b, lam

# def mixup_criterion(y_a, y_b, lam):
#     return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# %%
class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)


# %%
import colorednoise as cn

class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


# %%
class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_steps=5, sr=32000):
        super().__init__(always_apply, p)

        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        augmented = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return augmented


# %%
class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in ["uniform", "fade", "fade", "cosine", "sine"],             "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit= db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented


# %%
def horizontal_flip(img):
    horizontal_flip_img = img[:, ::-1]
    return horizontal_flip_img

def vertical_flip(img):
    vertical_flip_img = img[::-1, :]
    return vertical_flip_img

def addNoisy(img):
    noise_img = util.random_noise(img)
    return noise_img

def contrast_stretching(img):
    contrast_img = exposure.rescale_intensity(img)
    return contrast_img

def randomGaussian(img):
    gaussian_img = gaussian(img)
    return gaussian_img

def grayScale(img):
    gray_img = rgb2gray(img)
    return gray_img

def randomGamma(img):
    img_gamma = exposure.adjust_gamma(img)
    return img_gamma

def nonAug(img):
    return img

# %% [markdown]
# ## Transform

# %%
# transforms
train_transform = transforms.Compose([
  PinkNoiseSNR(min_snr=10, always_apply=False, p=0.5),
  # PitchShift(max_steps=2, sr=params.sr, always_apply=False, p=0.3),
  # VolumeControl(mode="sine", always_apply=False, p=0.3)
])
valid_transform = transforms.Compose([
    # transforms.CenterCrop((config.mel, config.CLIP_LEN)),
    # transforms.ToTensor()
])
label_transform = transforms.Compose([
    # transforms.ToTensor()
])


# %%



# %%



# %%


# %% [markdown]
# ## Dataset

# %%
# Data load
df_train_tp = pd.read_csv(config.TRAIN_TP_CSV)

# create dictionary
dic_rec_spec = {}
for index, row in df_train_tp.iterrows():
    if row["recording_id"] not in dic_rec_spec:
        dic_rec_spec[row["recording_id"]] = [row["species_id"]]
    else:
        dic_rec_spec[row["recording_id"]].append(row["species_id"])
dic_rec_spec["77299bde7"]


# %%
# add column per birds and flogs
for col in range(24):
    df_train_tp[col] = 0.

# one-hot encoding
for index, row in df_train_tp.iterrows():
    specId = row["species_id"]
    df_train_tp.iloc[index, df_train_tp.columns.get_loc(specId)] = 1

    for duplicateSpecId in dic_rec_spec[row["recording_id"]]:
        if specId != duplicateSpecId:
            df_train_tp.iloc[index, df_train_tp.columns.get_loc(duplicateSpecId)] = 1 * config.MIX_LABEL

# grouping
# df_train_tp = df_train_tp.groupby("recording_id", as_index=False).max()

# check
print(len(df_train_tp))
display(df_train_tp[df_train_tp["recording_id"] == "77299bde7"].head())


# %%
# display(df_train_tp[df_train_tp["recording_id"] == "00b404881"].head())
# df_train_tp_grouped = df_train_tp.groupby(["species_id", "recording_id"], as_index=False).max()
# display(df_train_tp_grouped[df_train_tp_grouped["recording_id"] == "00b404881"])


# %%
# load data
ids = []
specIds = []
record_ids = []
labels = []
offsets = []
for index, row in df_train_tp.iterrows():
    ids.append(row.values[0])
    specIds.append(row.values[1])
    record_ids.append(row.values[2])
    labels.append(row.values[6:30])
    offsets.append(row.values[5])

labels = np.array(labels).astype(float)

print('id', ids[584])
print('specid', specIds[584])
print('recid', record_ids[584])
print('label', labels[584])
print('label shape', labels[584].shape)
print('id len', len(ids))
print('offset', offsets[584])
print('offset', offsets[584] / params.sr)


# %%
# class RainforestDatasets(torch.utils.data.Dataset):
#     def __init__(self, _transform, train = True):
#         self.transform = _transform
#         self.train = train

#         # data load
#         self.labelset = labels
#         self.dataset = []
#         for id in ids:
#             # read npy
#             melspec = torch.load(os.path.join(config.TRAIN_AUDIO_ROOT, str(id) + ".pt")) # (dim, seq_len)
#             # melspec = torch.from_numpy(melspec)
#             # melspec = melspec.unsqueeze(0) # add channel for first convolution
#             # melspec = melspec[np.newaxis, :, :] # add channel for first convolution
#             self.dataset.append(melspec)

#         self.dataset = np.array(self.dataset).astype(float)
#         self.datanum = len(self.dataset)
        

#     def __len__(self):
#         return self.datanum

#     def __getitem__(self, idx):
#         # get data
#         out_label = self.labelset[idx]
#         out_data = self.dataset[idx]

#         # to tensor
#         out_label = torch.from_numpy(out_label).float()
#         # out_data = torch.from_numpy(out_data).float()
        
#         # transform label
#         # out_data = self.transform(out_data)
#         # out_label = label_transform(out_label)

#         # out_data = out_data.transpose(0, 1) # (dim, seq_len) => (seq_len, dim)
#         # out_data = out_data.unsqueeze(0) # add channel for first convolution (seq_len, dim) => (c, seq_len, dim)
#         # out_data = torch.stack([out_data, out_data, out_data]) # add channel for first convolution (seq_len, dim) => (c, seq_len, dim)
#         # print(type(out_data))
#         # print(type(np.array(out_label)))
#         # print(out_data.shape)

#         # # encoding on cpu(Its important for reduce usage of gpu memory)
#         # out_data = out_data.unsqueeze(0) # add fake batch
#         # out_data = model_efn.extract_features(out_data)
#         # out_data = out_data[0]

#         return out_data, out_label


# %%
np.ndarray.argmax(labels[1])


# %%
# in: idx, out: batch(valid), label(s)
class RainforestTrainDatasets(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = labels
        self.ids = ids     
        self.datanum = len(labels)   
        self.record_ids = record_ids
        self.offsets = offsets
        self.augs = [
            addNoisy,
            contrast_stretching,
            randomGaussian, 
            randomGamma,
            nonAug
        ]

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        # get data
        out_label = self.labels[idx]

        # random crop
        randomCropOffset = int((int(np.random.rand() * self.offsets[idx])))

        # load wav
        wavnp = np.load(Path('../input//rfcx-species-audio-detection/train_mel/' + str(self.ids[idx]) + '.npy'))

        # transform
        # wavnp = train_transform(wavnp)
        
        if randomCropOffset >= 0:
            wavnp = wavnp[0 + randomCropOffset: (10 * params.sr) + randomCropOffset]
        else:
            wavnp = wavnp[len(wavnp) - (10 * params.sr) + randomCropOffset : len(wavnp) + randomCropOffset]
        wavnp = wav2mel(wavnp) # 10s clipping

        # aug= random.choice(self.augs)
        # wavnp = aug(wavnp)

        # dim, seq_len => seq_len, dim
        wavnp = wavnp.T

        # add channel
        wavnp = np.stack([wavnp, wavnp, wavnp])

        # to Tensor
        wavTensor = torch.from_numpy(wavnp).float()

        return wavTensor, out_label
        # return wavTensor, np.ndarray.argmax(out_label)


# %%
# in: idx, out: batch(valid), label(s)
class RainforestValidDatasets(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = labels
        self.ids = ids     
        self.datanum = len(labels)   
        self.record_ids = record_ids
        self.offsets = offsets

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        # get data
        out_label = self.labels[idx]

        # random crop
        randomCropOffset = int((int(np.random.rand() * self.offsets[idx])))

        # load wav
        wavnp = np.load(Path('../input//rfcx-species-audio-detection/train_mel/' + str(self.ids[idx]) + '.npy'))
        if randomCropOffset >= 0:
            wavnp = wavnp[0 + randomCropOffset: (10 * params.sr) + randomCropOffset]
        else:
            wavnp = wavnp[len(wavnp) - (10 * params.sr) + randomCropOffset : len(wavnp) + randomCropOffset]
        wavnp = wav2mel(wavnp) # 10s clipping

        # dim, seq_len => seq_len, dim
        wavnp = wavnp.T

        # add channel
        wavnp = np.stack([wavnp, wavnp, wavnp])

        # to Tensor
        wavTensor = torch.from_numpy(wavnp).float()

        return wavTensor, out_label
        # return wavTensor, np.ndarray.argmax(out_label)


# %%
# # in: idx, out: batch(valid), label(s)
# class RainforestValidDatasets(torch.utils.data.Dataset):
#     def __init__(self):
#         self.labels = labels
#         self.ids = ids     
#         self.datanum = len(labels)   

#     def __len__(self):
#         return self.datanum

#     def __getitem__(self, idx):
#         # get data
#         out_label = self.labels[idx]

#         outdatas = []

#         #TODO: 11 is magic number, due to change it to conf
#         for i in range(11):
#             if i % 3 == 0:
#                 # load melspec(dim, seq_len)
#                 melspec =  np.load(os.path.join(config.VALID_AUDIO_ROOT, str(self.ids[idx]) + "_" + str(i) + ".npy"))
#                 # add channel
#                 melspec = np.stack([melspec.T, melspec.T, melspec.T])
#                 outdatas.append(melspec)

#         # list 2 tochtensor(batch, channel, seq_len, dim)
#         outdatas = torch.from_numpy(np.array(outdatas))

#         return outdatas, out_label

# %% [markdown]
# ## Check Data

# %%
# # ds train
# import librosa.display
# ds = RainforestTrainDatasets(train_transform)
# loader = DataLoader(ds)

# # check aug
# for x, y in loader:
#     a = 1

# fig, ax = plt.subplots(figsize=(15, 5))
# ax.set(title='train random crop')
# img = librosa.display.specshow(
#     x.numpy()[0][0].T, 
#     sr=48000,
#     x_axis='time', 
#     y_axis='linear', 
#     ax=ax)


# ds = RainforestDatasets(valid_transform)
# loader = DataLoader(ds)

# # check aug
# for x, y in loader:
#     a = 1

# fig, ax = plt.subplots(figsize=(15, 5))
# ax.set(title='validation center crop')
# img = librosa.display.specshow(
#     x.numpy()[0][0].T, 
#     sr=48000,
#     x_axis='time', 
#     y_axis='linear', 
#     ax=ax)

# # ax.set(title=f'Mel-frequency spectrogram of {row["recording_id"]}')
# # fig.colorbar(img, ax=ax, format="%+2.f dB")
# print(x.numpy()[0][0].T.shape)

# melspec = np.load(os.path.join(config.TRAIN_AUDIO_ROOT, str(1215) + ".npy"))
# fig, ax = plt.subplots(figsize=(15, 5))
# img = librosa.display.specshow(
#     melspec, 
#     sr=48000,
#     x_axis='time', 
#     y_axis='linear', 
#     ax=ax)
# print(melspec.shape)

# %% [markdown]
# ## Modeling

# %%
# # Conformer
# # https://arxiv.org/abs/2005.08100
# class RainforestTransformer(nn.Module):
#     def __init__(self):
#         super(RainforestTransformer, self).__init__()         

#         self.encoding = model_efn
#         # self.pointwise = nn.Conv2d(config.ENC_CH, 1, (1, 1))
#         self.conv = nn.Conv2d(config.ENC_CH, 1, (config.KERNEL_SIZE_SEQ, config.KERNEL_SIZE), stride=config.KERNEL_STRIDE)
#         self.linear = nn.Linear(int((((((config.ENC_DIM - config.KERNEL_SIZE) / config.KERNEL_STRIDE) + 1) - config.POOL_SIZE) / config.POOL_STRIDE) + 1), config.ENC_DIM)
#         self.dropout = nn.Dropout(config.DROPOUT)
        
#         self.conformerblock = ConformerBlock(
#             dim = config.ENC_DIM,
#             dim_head = 64,
#             heads = 8,
#             ff_mult = 4,
#             conv_expansion_factor = 2,
#             conv_kernel_size = 31,
#             attn_dropout = config.DROPOUT,
#             ff_dropout = config.DROPOUT,
#             conv_dropout = config.DROPOUT
#         )
#         self.conformerblock2 = ConformerBlock(
#             dim = config.ENC_DIM,
#             dim_head = 64,
#             heads = 8,
#             ff_mult = 4,
#             conv_expansion_factor = 2,
#             conv_kernel_size = 31,
#             attn_dropout = config.DROPOUT,
#             ff_dropout = config.DROPOUT,
#             conv_dropout = config.DROPOUT
#         )

#         self.decoder = nn.Linear(1 * int((((((config.ENC_LEN - config.KERNEL_SIZE_SEQ) / config.KERNEL_STRIDE) + 1) -  config.POOL_SIZE) / config.POOL_STRIDE) + 1) * config.ENC_DIM, config.NUM_BIRDS)

#         # devided by stride
    
#     # x: (b, c, seqlen, dim)
#     def forward(self, x):
#         # (b, c, seqlen, dim) => (b, c, seqlen, dim)
#         x = self.encoding.extract_features(x)
#         # enc = self.pointwise(enc)

#         # (b, c, seqlen, dim) <= encoded matrix
#         # point-wise convokution for convolution channel.
#         h = F.relu(self.conv(x))
#         h = F.max_pool2d(h, config.POOL_SIZE, stride=config.POOL_STRIDE)
#         h = self.linear(h)
#         h = h.transpose(0, 1)[0] # transpose batch and channel to delet channel dimension
#         h = self.conformerblock(h)
#         h = self.conformerblock2(h)
#         # h = self.conformerblock3(h)
#         # h = self.conformerblock4(h)
#         # h = self.conformerblock5(h)
#         # h = self.conformerblock6(h)
#         h = h.view(-1, 1 * int((((((config.ENC_LEN - config.KERNEL_SIZE_SEQ) / config.KERNEL_STRIDE) + 1) -  config.POOL_SIZE) / config.POOL_STRIDE) + 1) * config.ENC_DIM)
#         out = self.decoder(h)
#         return out


# %%



# %%
temp_img = torch.from_numpy(np.random.randn(5, 2))
temp_ch = torch.stack([temp_img, temp_img, temp_img])
temp_ch.shape
temp_batch = torch.stack([temp_ch, temp_ch, temp_ch, temp_ch, temp_ch, temp_ch])
print(temp_batch.shape)

torch.mean(temp_batch, dim=3).shape

# %% [markdown]
# ## Visualize Model

# %%
# # dummy = torch.stack([clip, clip, clip]).unsqueeze(0)
# model = RainforestTransformer()
# y = model(sample.unsqueeze(0))
# make_dot(y,params=dict(model.named_parameters()))

# %% [markdown]
# ## Preserve Memory

# %%
# # delete unusual var
del sample
# del model_efn
# del y

# %% [markdown]
# ## Metric

# %%
# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indiscating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# %%
allclip = np.stack([
    np.zeros([3,10,2]),
    np.ones([3,10,2]),
    np.zeros([3,10,2]),
    np.ones([3,10,2]),
    np.zeros([3,10,2]),
])
# div, c, seq_len, dim
print(allclip.shape)

# b, d, c, s, d
batch = np.stack([allclip, allclip, allclip, allclip])
print(batch.shape)

# reshape
print('loop all divnum')
for i in range(batch.shape[1]):
    print(batch[:, i, :, :].shape)
    # print(batch[])
    print(batch[:, i, :, :].sum())
    # 4 * 3 * 10 * 2


# %%
a = np.random.randint(0, 10, (2,4,3))
print(a)

print(np.max(a, axis=0))
# print(np.max(a, axis=1))

# print(np.max(a, axis=2))


# %%
torch.max(torch.Tensor([0,1,2]), dim=0).values

# %% [markdown]
# ## Custom Optimizer

# %%
class Adas(Optimizer):
    r"""
    Introduction:
        For the mathematical part see https://github.com/YanaiEliyahu/AdasOptimizer,
        the `Theory` section contains the major innovation,
        and then `How ADAS works` contains more low level details that are still somewhat related to the theory.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr: float > 0. Initial learning rate that is per feature/input (e.g. dense layer with N inputs and M outputs, will have N learning rates).
        lr2: float >= 0.  lr's Initial learning rate. (just ~1-2 per layer, additonal one because of bias)
        lr3: float >= 0. lr2's fixed learning rate. (global)
        beta_1: 0 < float < 1. Preferably close to 1. Second moments decay factor to update lr and lr2 weights.
        beta_2: 0 < float < 1. Preferably close to 1. 1/(1 - beta_2) steps back in time that `lr`s will be optimized for, larger dataset might require more nines.
        beta_3: 0 < float < 1. Preferably close to 1. Same as beta_2, but for `lr2`s.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
    """

    def __init__(self, params,
            lr = 0.001, lr2 = .005, lr3 = .0005,
            beta_1 = 0.999, beta_2 = 0.999, beta_3 = 0.9999,
            epsilon = 1e-8, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid lr: {}".format(lr))
        if not 0.0 <= lr2:
            raise ValueError("Invalid lr2: {}".format(lr))
        if not 0.0 <= lr3:
            raise ValueError("Invalid lr3: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta_1 parameter: {}".format(betas[0]))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta_2 parameter: {}".format(betas[1]))
        if not 0.0 <= beta_3 < 1.0:
            raise ValueError("Invalid beta_3 parameter: {}".format(betas[2]))
        defaults = dict(lr=lr, lr2=lr2, lr3=lr3, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, epsilon=epsilon)
        self._varn = None
        self._is_create_slots = None
        self._curr_var = None
        self._lr = lr
        self._lr2 = lr2
        self._lr3 = lr3
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._beta_3 = beta_3
        self._epsilon = epsilon
        super(Adas, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adas, self).__setstate__(state)

    @torch.no_grad()
    def _add(self,x,y):
        x.add_(y)
        return x

    @torch.no_grad()
    # TODO: fix variables' names being too convoluted in _derivatives_normalizer and _get_updates_universal_impl
    def _derivatives_normalizer(self,derivative,beta):
        steps = self._make_variable(0,(),derivative.dtype)
        self._add(steps,1)
        factor = (1. - (self._beta_1 ** steps)).sqrt()
        m = self._make_variable(0,derivative.shape,derivative.dtype)
        moments = self._make_variable(0,derivative.shape,derivative.dtype)
        m.mul_(self._beta_1).add_((1 - self._beta_1) * derivative * derivative)
        np_t = derivative * factor / (m.sqrt() + self._epsilon)
        #the third returned value should be called when the moments is finally unused, so it's updated
        return (moments,np_t,lambda: moments.mul_(beta).add_((1 - beta) * np_t))

    def _make_variable(self,value,shape,dtype):
        self._varn += 1
        name = 'unnamed_variable' + str(self._varn)
        if self._is_create_slots:
            self.state[self._curr_var][name] = torch.full(size=shape,fill_value=value,dtype=dtype,device=self._curr_var.device)
        return self.state[self._curr_var][name]

    @torch.no_grad()
    def _get_updates_universal_impl(self, grad, param):
        lr = self._make_variable(value = self._lr,shape=param.shape[1:], dtype=param.dtype)
        moment, deriv, f = self._derivatives_normalizer(grad,self._beta_3)
        param.add_( - torch.unsqueeze(lr,0) * deriv)
        lr_deriv = torch.sum(moment * grad,0)
        f()
        master_lr = self._make_variable(self._lr2,(),dtype=torch.float32)
        m2,d2, f = self._derivatives_normalizer(lr_deriv,self._beta_2)
        self._add(lr,master_lr * lr * d2)
        master_lr_deriv2 = torch.sum(m2 * lr_deriv)
        f()
        m3,d3,f = self._derivatives_normalizer(master_lr_deriv2,0.)
        self._add(master_lr,self._lr3 * master_lr * d3)
        f()

    @torch.no_grad()
    def _get_updates_universal(self, param, grad, is_create_slots):
        self._curr_var = param
        self._is_create_slots = is_create_slots
        self._varn = 0
        return self._get_updates_universal_impl(grad,self._curr_var.data)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adas does not support sparse gradients')
                self._get_updates_universal(p,grad,len(self.state[p]) == 0)
        return loss


# %%


# %% [markdown]
# ## Train

# %%
def train():

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    # msss = MultilabelStratifiedShuffleSplit(n_splits=config.N_FOLDS, test_size=config.TEST_SIZE, random_state=config.SEED)

    # Read dataset
    train_datasets = RainforestTrainDatasets()
    # valid_datasets = RainforestDatasets(_transform=valid_transform)

    # valid dataset dosen't need to transform(already be croped)
    valid_datasets = RainforestValidDatasets()

    best_epochs = []
    best_lwlraps = []

    # tensorboard
    # if IS_WRITRE_LOG:
    #     writer = SummaryWriter(log_dir="./logs/" + EXP_NAME)

    # k-fold
    # for kfoldidx, (train_index, valid_index) in enumerate(msss.split(labels, labels)):
    for kfoldidx, (train_index, valid_index) in enumerate(skf.split(specIds, specIds)):

        # # model 
        # model = RainforestTransformer()
        # model.to(device)
        # model = EfficientNet.from_pretrained('efficientnet-b0')
        model = get_model()
        # num_ftrs = model._fc.in_features
        # model._fc = nn.Linear(num_ftrs, config.NUM_BIRDS)
        model.to(device)

        np.save('train_index_fold_' + str(kfoldidx), np.array(train_index))
        np.save('valid_index_fold_' + str(kfoldidx), np.array(valid_index))

        # init
        best_lwlrap = 0.
        best_epoch = 0

        if IS_WRITRE_LOG:
            run = wandb.init(config=config, project=PROJECT, group=EXP_NAME, reinit=True)
            print('wandb init')
            wandb.run.name = EXP_NAME + '-fold-' + str(kfoldidx)
            wandb.run.save()
            wandb.watch(model)

        # criterion
        print('wandb init2')
        criterion = nn.BCEWithLogitsLoss().cuda()
        # criterion = nn.CrossEntropyLoss().cuda()

        # optimizer
        # optimizer = Adam(params=model.parameters(), lr=config.lr, amsgrad=False)
        optimizer = toptim.RAdam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        # optimizer = Adas(model.parameters(), lr=config.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        # print(optimizer)

        # train
        train_subset = Subset(train_datasets, train_index)
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_NUM, shuffle=True)

        # validation
        valid_subset = Subset(valid_datasets, valid_index)
        valid_loader = DataLoader(valid_subset, batch_size=config.VALID_BATCH_NUM, shuffle=False)

        # # scheduler
        # # scheduler = CosineAnnealingLR(optimizer, T_max=config.t_max, eta_min=config.eta_min)
        # num_train_steps = int(len(train_loader) * config.EPOCH_NUM)
        # num_warmup_steps = int(0.1 * config.EPOCH_NUM * len(train_loader))
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

        # epoch
        mb = master_bar(range(config.EPOCH_NUM))
        mb.names = ['avg_loss', 'avg_val_loss', 'lwlrap']

        # Epoch
        for epoch in mb:

            # start time
            start_time = time.time()

            # train mode
            model.train()

            # init loss
            avg_loss = 0.

            # batch training
            train_batch_preds = []
            train_batch_labels = []
            for x_batch, y_batch in progress_bar(train_loader, parent=mb):

                # MixUp
                dice = np.random.rand(1)
                if dice < config.MIXUP_PROB:
                    # mixup
                    x_batch, y_batch, y_batch_b, lam = mixup_data(x_batch, y_batch, alpha=config.MIXUP, use_cuda=True)

                # spec Aug
                dice_s = np.random.rand(1)
                if dice_s < config.SPEC_PROB:
                    # specaug
                    x_batch = spec_augmenter(x_batch)                

                # forward
                preds = model(x_batch.to(device))

                if dice < config.MIXUP_PROB:
                    loss = mixup_criterion(criterion, preds, y_batch.to(device), y_batch_b.to(device), lam)
                else:
                    loss = criterion(preds, y_batch.to(device)) # It dosen't need Sigmoid, because BCE includes sigmoid function.

                # loss = criterion(preds, y_batch.to(device))
                # loss = criterion(preds, torch.max(y_batch, dim=1).indices.to(device, dtype=torch.long)) # It dosen't need Sigmoid, because BCE includes sigmoid function.

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / len(train_loader)

                del loss

                # add preds
                train_batch_preds.extend(torch.sigmoid(preds).detach().cpu().numpy().tolist())
                train_batch_labels.extend(y_batch.detach().cpu().numpy().tolist())

            # calc score
            score, weight = calculate_per_class_lwlrap(np.array(train_batch_labels), np.array(train_batch_preds))
            train_lwlrap = (score * weight).sum()

            # last_preds =  np.array(train_batch_preds)
            # last_labels = np.array(train_batch_labels)

            # validation mode
            model.eval()
            valid_batch_preds = []
            valid_batch_labels = []
            # valid_preds = np.zeros((len(valid_index), config.NUM_BIRDS))
            avg_val_loss = 0.

            # validation
            # for x_batch, y_batch in progress_bar(train_loader, parent=mb):
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                # !!!caution!!!
                # x_batch's shape (batch, devide length(i.e. 51), channel, seq_len, dim)
                # extract column

                # wholeclip_preds = []
            
                # for divnum in range(x_batch.shape[1]):
                #     x_batch_divided = x_batch[:, divnum, :, :]
                #     preds = model(x_batch_divided.to(device)).detach() # (batch, species_id)
                #     wholeclip_preds.append(preds.cpu().numpy().tolist()) # (divnum, batch, species_id)

                # # get max via divnum
                # # (batch, preds_dimention)
                # preds = torch.max(torch.from_numpy(np.array(wholeclip_preds)).float(), dim=0).values
                preds = model(x_batch.to(device)).detach() # (batch, species_id)

                # preds = model(x_batch.to(device)).detach()
                loss = criterion(preds.to(device), y_batch.to(device))
                # loss = criterion(preds.to(device), torch.max(y_batch, dim=1).indices.to(device, dtype=torch.long))

                preds = torch.sigmoid(preds)
                # valid_preds[i * config.VALID_BATCH_NUM: (i+1) * config.VALID_BATCH_NUM] = preds.cpu().numpy()
                avg_val_loss += loss.item() / len(valid_loader)

                valid_batch_preds.extend(preds.detach().cpu().numpy().tolist())
                valid_batch_labels.extend(y_batch.detach().cpu().tolist())

            # calc score
            # score, weight = calculate_per_class_lwlrap(labels[valid_index], valid_preds)
            score, weight = calculate_per_class_lwlrap(np.array(valid_batch_labels), np.array(valid_batch_preds))
            lwlrap = (score * weight).sum()

            # update lr
            # scheduler.step()
            # scheduler.step(avg_val_loss)

            # tensorboard
            if IS_WRITRE_LOG:
                # tensorboard
                # writer.add_scalar("train_loss/fold-" + str(kfoldidx), avg_loss, epoch)
                # writer.add_scalar("valid_loss/fold-" + str(kfoldidx), avg_val_loss, epoch)
                # writer.add_scalar("train_lwlrap/fold-" + str(kfoldidx), train_lwlrap, epoch)
                # writer.add_scalar("valid_lwlrap/fold-" + str(kfoldidx), lwlrap, epoch)

                wandb.log({
                    'loss/train': avg_loss,
                    'lwlrap/train': train_lwlrap,
                    'loss/validation': avg_val_loss,
                    'lwlrap/validation': lwlrap,
                    'epoch': epoch
                })


            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} train_lwlrap: {train_lwlrap:.6f}  val_lwlrap: {lwlrap:.6f}  time: {elapsed:.0f}s')
        
            if lwlrap > best_lwlrap and epoch > 10:
                best_epoch = epoch + 1
                best_lwlrap = lwlrap
                # torch.save(model.state_dict(), 'weight_best_' + str(EXP_NUM) + '_fold' + str(kfoldidx) +'.pt')
                torch.save(model.state_dict(), 'weight_best_fold' + str(kfoldidx) +'.pt')
                np.save('train_batch_preds_' + str(kfoldidx), np.array(train_batch_preds))
                np.save('train_batch_labels_' + str(kfoldidx), np.array(train_batch_labels))
                np.save('valid_batch_preds_' + str(kfoldidx), np.array(valid_batch_preds))
                np.save('valid_batch_labels_' + str(kfoldidx), np.array(valid_batch_labels))
            
        best_epochs.append(best_epoch)
        best_lwlraps.append(best_lwlrap)

        # return last_preds, last_labels

        if IS_WRITRE_LOG:
            run.finish()

    # if IS_WRITRE_LOG:
    #     writer.close()
    
    return {
        'best_epoch': best_epochs,
        'best_lwlrap': best_lwlraps,
    }



# %%
result = train()
print(result)

# %% [markdown]
# ### Folds Analytics

# %%
# calc lwlrap
train_batch_labels = np.load("./train_batch_labels.npy")
train_batch_preds = np.load("./train_batch_preds.csv.npy")
# extrct under < 1.0
valid_batch_labels = np.load("./valid_batch_labels.npy")
valid_batch_preds = np.load("./valid_batch_preds.csv.npy")


# 
print(train_batch_labels.shape)
print(train_batch_labels[0])

print(train_batch_preds.shape)
print(train_batch_preds[0])

print(valid_batch_labels.shape)
print(valid_batch_labels[0])

print(valid_batch_preds.shape)
print(valid_batch_preds[0])


# %%



# %%


# %% [markdown]
# ## Submission

# %%
# prediction
models = []
for fold in range(config.N_FOLDS):
   #  if not fold == 0:
    # load network
    print(fold)
   #  model = RainforestTransformer()
   #  model = EfficientNet.from_pretrained('efficientnet-b0')
    model = get_model()
    # num_ftrs = model._fc.in_features
    # model._fc = nn.Linear(num_ftrs, config.NUM_BIRDS)

    # torch.save(model.state_dict(), 'weight_best_' + str(EXP_NUM) + '_fold' + str(kfoldidx) +'.pt')
    model.load_state_dict(torch.load('weight_best_fold' + str(fold) +'.pt'))
    # print('weight_best_' + str(EXP_NUM) + '_fold' + str(fold) +'.pt')
    # model.load_state_dict(torch.load('weight_best_' + str(EXP_NUM) + '_fold' + str(fold) +'.pt'))
    model.to(device)
    model.eval()
    models.append(model)


# %%
# write submission
with open('submission_' + EXP_NAME + '.csv', 'w', newline='') as csvfile:
# with open('submission_' + EXP_NAME + '_sum.csv', 'w', newline='') as csvfile:
    print('submission_' + EXP_NAME + '.csv')
    submission_writer = csv.writer(csvfile, delimiter=',')
    submission_writer.writerow(['recording_id','s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
                               's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23'])
    
    test_files = os.listdir(config.TEST_AUDIO_FLAC)
    
    # Every test file is split on several chunks and prediction is made for each chunk
    for i in tqdm(range(0, len(test_files))):
    # for i in range(0, 1):
        # read data
        # X_test = torch.from_numpy(np.load(os.path.join(config.TEST_AUDIO_ROOT, test_files[i])))

        # (dim, seq_len)        
        # X_test = np.load(os.path.join(config.TEST_AUDIO_ROOT, test_files[i]))
        X_test_batch = []

        # muptiply number
        # TODO: 51 is magic number. You have to rewrite 51 to vars.
        dev_num = 6

        # Cutting!
        for idx in range(dev_num):
            recId =  test_files[i].split('.')[0]
            X_test = np.load(os.path.join(config.TEST_AUDIO_ROOT, recId + '_' + str(idx) + '.npy')) # (DIM, seq_len)
            X_test_clip = X_test.T # (seq_len, DIM)
            # X_test_clip = X_test_clip[np.newaxis, :, :] # add fake channel
            X_test_clip = np.stack([X_test_clip, X_test_clip, X_test_clip]) # expand to channel
            X_test_batch.append(X_test_clip.tolist())

        # to_tensor
        X_test_batch = torch.from_numpy(np.array(X_test_batch)).float() # (batch, channel, seq_len, dim)
        X_test_batch = X_test_batch.to(device)

        # predict
        output_list = []
        for m in models:
            outputs = []
            for x_b in X_test_batch:
                output = m(torch.stack([x_b]))
                outputs.append(output[0].cpu().detach().numpy().tolist())
            # outputs S= m(X_test_batch)
            maxed_output = torch.max(torch.from_numpy(np.array(outputs)).float(), dim=0) # max about batch clips
            # maxed_output = torch.sum(torch.from_numpy(np.array(outputs)).float(), dim=0) # max about batch clips
            # maxed_output = torch.max(outputs, dim=0) # max about batch clips
            maxed_output = maxed_output.values.cpu().detach()
            # maxed_output = maxed_output.cpu().detach()
            output_list.append(maxed_output)
        avg_maxed_output = torch.mean(torch.stack(output_list), dim=0)
        
        file_id = str.split(test_files[i], '.')[0]
        write_array = [file_id]
        
#         for out in maxed_output:
        for out in avg_maxed_output:
            write_array.append(out.item())
    
        submission_writer.writerow(write_array)
        
        if i % 100 == 0 and i > 0:
            print('Predicted for ' + str(i) + ' of ' + str(len(test_files) + 1) + ' files')

        
print('finished!')


# %%
maxed_output


# %%
print(9)


# %%
target = torch.empty(3, dtype=torch.long).random_(5)
target


# %%
input = torch.randn(3, 5, requires_grad=True)
input


# %%
a = torch.randn(3, 5)
print(a)
torch.max(a, dim=1).indices


# %%



