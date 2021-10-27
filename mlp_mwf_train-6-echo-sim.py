import torch
import pytorch_lightning as pl
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pylab as plt
#import matplotlib as matp
import numpy as np
from Hang.utils_u_groupnorm_pytorchLightning import *
from utils import *
from torch.utils import data
from numpy import zeros
import time as time
import pdb
import nibabel as nib
import torchvision.transforms
import random

np.random.seed(0)
random.seed(5)
torch.manual_seed(0)

numbers = ["0001","0017","0018","0038","0040","0042","0046","0087","0090","0108","0116","0131","0178","0190",
           "0227","0248","0267","0282","0285","0398","0448","0466","0504","0514","0535","0564","0598","0606",
           "0607","0618","0620","0623","0642","0646","0655","0668","0675","0681","0719","0761","0762","0783",
           "0786","0868","0877","0887","0895","0902","0931","0979","1007","1013","1029","1033","1068","1142",
           "1143","1163","1190","1260","1275","1347","1383","1389","1416","1435","1441","1447","1451","1514",
           "1520","1602","1611","1621","1680","1684","1686","1710","1720","1739","1743","1749","1753","1760",
           "1795","1805","1845","1858","1876","1889","1892","1898","1899","1918","1924","1932","1952","1961",
           "1972","1987","2003","2007","2016","2020","2022","2030","2045","2047","2049","2053","2055","2074",
           "2077","2080","2091","2094","2103","2115","2128","2142","2144","2146","2152","2156","2158","2160",
           "2161","2179","2180","2181","2183","2186","2188","2212","2221","2231","2234","2245"]

fastt2_files = []
mask_files = []
mwf_files = []

SPLIT = 50
VAL = 25
numbers = numbers[:SPLIT+VAL]

for number in numbers:
    file_root = "../" + number + "/"
    mask_files.append(file_root + "tightmask.nii.gz")
    fastt2_files.append(file_root + "FASTT2_SNR_200_SIM_6_ECHO.nii.gz")
    mwf_files.append(file_root + "MWF.nii.gz")

mask_array = []
brains = []
labels = []
    
for i in range(0,len(mask_files)):
    mask_array.append(nib.load(mask_files[i]).get_fdata()) 

for i in range(0,len(mask_files)):
    brain = nib.load(fastt2_files[i]).get_fdata().transpose((3,0,1,2))
    brains.append(brain)
        
for i in range(0,len(mask_files)):
    mwf = nib.load(mwf_files[i]).get_fdata() * mask_array[i]
    mwf = mwf[None] 
    labels.append(mwf)
    
labels = np.array(labels)
brains = np.array(brains)
mask_array = np.array(mask_array)

trainLabels = labels[:SPLIT]
trainBrains = brains[:SPLIT]
trainMaskArray = mask_array[:SPLIT]

validLabels = labels[SPLIT:SPLIT+VAL]
validBrains = brains[SPLIT:SPLIT+VAL]
validMaskArray = mask_array[SPLIT:SPLIT+VAL]

flattenedLabels = []
flattenedBrains = []
for i in range(6):
    flattenedBrains.append(trainBrains[:,i][trainMaskArray == 1])
for i in range(1):
    flattenedLabels.append(trainLabels[:,i][trainMaskArray == 1])
flattenedLabels = np.swapaxes(np.array(flattenedLabels), 0, 1)
flattenedBrains = np.swapaxes(np.array(flattenedBrains), 0, 1)
trainLabels = flattenedLabels
trainBrains = flattenedBrains

flattenedLabels = []
flattenedBrains = []
for i in range(6):
    flattenedBrains.append(validBrains[:,i][validMaskArray == 1])
for i in range(1):
    flattenedLabels.append(validLabels[:,i][validMaskArray == 1])
flattenedLabels = np.swapaxes(np.array(flattenedLabels), 0, 1)
flattenedBrains = np.swapaxes(np.array(flattenedBrains), 0, 1)
validLabels = flattenedLabels
validBrains = flattenedBrains

def calculate_stats(idx):
    mean = trainLabels[:,idx].mean()
    std = trainLabels[:,idx].std()
    return mean, std

stats = []
for i in range(1):
    stats.append(calculate_stats(i))

for i in range(len(stats)):
    trainLabels[:,i] -= stats[i][0]
    trainLabels[:,i] /= stats[i][1]

for i in range(len(stats)):
    validLabels[:,i] -= stats[i][0]
    validLabels[:,i] /= stats[i][1]

for i in range(len(stats)):
    print(stats[i][0], stats[i][1])
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std_range = (0,std) if (type(std) == float or type(std) == int) else (std[0],std[1])
        self.mean = mean
    def __call__(self, tensor):
        std = random.random() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]
        return tensor + torch.randn(tensor.size()) * std + self.mean
    
class Dataset_Generator(Dataset):
    def __init__(self, brains, labels, transforms = None):
        self.brains = brains.astype(float)
        self.labels = labels
        self.transforms = transforms
    def __len__(self):
        return len(self.brains)
    def __getitem__(self, idx):
        signal = self.brains[idx] / (self.brains[idx][0] + 1e-16)
        signal_label = self.labels[idx]
        if self.transforms is not None:
            idx = random.randrange(0,len(self.transforms))
            if (idx == 0):
                return (torch.tensor(signal.copy()).float(), torch.tensor(signal_label.copy()).float())
            else:
                signal = self.transforms[idx](torch.tensor(signal.copy()))
                signal_label = signal_label.copy()
                return (signal.float(), torch.tensor(signal_label.copy()).float())
        return (torch.tensor(signal.copy()).float(), torch.tensor(signal_label.copy()).float())

def dataset_and_dataloader_creator(data, label, shuffling, transforms = None):
    DS = Dataset_Generator(data, label, transforms)
    DL = DataLoader(DS, batch_size=512, shuffle=shuffling) #if is_deconv, bs = 8
    return DS,DL

transforms = [None, AddGaussianNoise(0., 0.02)]

TRAIN_DS, TRAIN_DL = dataset_and_dataloader_creator(trainBrains, trainLabels, True, transforms)
VALID_DS, VALID_DL = dataset_and_dataloader_creator(validBrains, validLabels, False)

print(len(TRAIN_DS))
print(len(VALID_DS))

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning.logging import TensorBoardLogger
from Hang.annPersonalBatchNorm_pytorchLightning import ann_256_256_32

model = ann_256_256_32(0.005, [256,256,32], decay_factor = 0.2, n_classes=1).float()
MODEL_TYPE = "256_256_32"
CHECKPOINT = False

filepath = f'../mlp_models/full_brain_{MODEL_TYPE}'
checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    save_top_k=True,
    verbose=True,
    monitor='avg_val_loss',
    mode='min',
    prefix=''
)

early_stop_callback = EarlyStopping(
   monitor='avg_val_loss',
   min_delta=0.00,
   patience=10,
   verbose=True,
   mode='min'
)

loggerName = f'ann_{MODEL_TYPE}'
logger = TensorBoardLogger("lightning_logs_ann_gpu3_256_256_32", name=loggerName)

if (CHECKPOINT):
    trainer = Trainer(max_epochs=300, gpus=[0], accumulate_grad_batches=2, 
                      logger=logger,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback, 
                      auto_lr_find=False, 
                      resume_from_checkpoint="../ann_gpu3_256_256_32/_ckpt_epoch_41.ckpt")
else:
    trainer = Trainer(max_epochs=300, gpus=[0], accumulate_grad_batches=2, 
                      logger=logger, 
                      checkpoint_callback=checkpoint_callback, 
                      early_stop_callback=early_stop_callback, 
                      auto_lr_find=False)

trainer.fit(model, TRAIN_DL, VALID_DL)
