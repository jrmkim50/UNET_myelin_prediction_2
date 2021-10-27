import torch
import pytorch_lightning as pl
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Hang.utils_u_groupnorm_pytorchLightning import *
from utils import *
import nibabel as nib
import torchvision.transforms
import random
import torchio as tio

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
    mask_files.append(file_root + "tightmask.nii.gz") #brain_mask.nii.gz
    fastt2_files.append(file_root + "FASTT2_FULL.nii.gz")
    mwf_files.append(file_root + "MWF.nii.gz")

mask_array = []
brains = []
labels = []

for i in range(0,len(mask_files)):
    # mask shape: 256, 256, 32
    mask_array.append(nib.load(mask_files[i]).get_fdata()) 

for i in range(0,len(mask_files)):
    brain = []
    for j in range(6):
        brain.append(nib.load(fastt2_files[i]).get_fdata()[:,:,:,j]  * mask_array[i])
    # brain size: 6, 256, 256, 32
    brains.append(brain)
        
for i in range(0,len(mask_files)):
    mwf = nib.load(mwf_files[i]).get_fdata() * mask_array[i]
    mwf = mwf[None]
    # mwf shape: 1, 256, 256, 32
    labels.append(mwf)

labels = np.array(labels)  # shape: n, 1, 256, 256, 32
brains = np.array(brains)  # shape: n, 6, 256, 256, 32
mask_array = np.array(mask_array)  # shape: n, 256, 256, 32

trainLabels = labels[:SPLIT]  # shape: SPLIT, 1, 256, 256, 32
trainBrains = brains[:SPLIT]  # shape: SPLIT, 6, 256, 256, 32
trainMaskArray = mask_array[:SPLIT]  # shape: SPLIT, 256, 256, 32

validLabels = labels[SPLIT:SPLIT+VAL]  # shape: n-SPLIT, 1, 256, 256, 32
validBrains = brains[SPLIT:SPLIT+VAL]  # shape: n-SPLIT, 6, 256, 256, 32
validMaskArray = mask_array[SPLIT:SPLIT+VAL]  # shape: n-SPLIT, 256, 256, 32

def calculate_stats(idx):
    mean = trainLabels[:,idx][trainMaskArray != 0].mean()
    std = trainLabels[:,idx][trainMaskArray != 0].std()
    return mean, std

stats = []
NUM_FEATURES = 1
for i in range(NUM_FEATURES):
    stats.append(calculate_stats(i))

for i in range(len(stats)):
    trainLabels[:,i] -= stats[i][0]
    trainLabels[:,i] /= stats[i][1]
    trainLabels[:,i][trainMaskArray == 0] = 0

for i in range(len(stats)):
    validLabels[:,i] -= stats[i][0]
    validLabels[:,i] /= stats[i][1]
    validLabels[:,i][validMaskArray == 0] = 0

for i in range(len(stats)):
    print(stats[i][0], stats[i][1])
    
np.save("stats_unet.npy", stats)

PATCH_SIZE = (128,128,32)
EXTRACTION_STEP = (10,10,1)

def generatePatches(volume, mask_array):
    patches_data = np.zeros((volume.shape[0],0)+PATCH_SIZE)
    patches = []
    # mask has shape of 256, 256, 32
    # volume has shape of n, 256, 256, 32 or 256, 256, 32 if it is a mask
    if (len(volume.shape) < 4):
        return extract_patches(mask_array,PATCH_SIZE,EXTRACTION_STEP)
    for j in range(len(volume)):
        # for each feature, extract patches
        patches_row = []
        patches_brain = extract_patches(volume[j],PATCH_SIZE,EXTRACTION_STEP)
        patches_mask = extract_patches(mask_array,PATCH_SIZE,EXTRACTION_STEP)
        for patch_num in range(len(patches_brain)):
            # check each patch. If it is within the mask, insert it into our patches arr
            if(patches_mask[patch_num].sum() > 0): 
                patches_row.append(patches_brain[patch_num])
        if (len(patches_row) > 0):
            # patches_row has shape n, 128, 128, 32
            patches.append(patches_row)
    # patches ends with shape len(volume), n, 128, 128, 32
    if (len(patches) > 0):
        patches_data = np.concatenate((patches_data,np.array(patches)), axis=1)
    # patches_data ends with shape n, len(volume), 128, 128, 32
    patches_data = np.swapaxes(patches_data, 0, 1)
    return patches_data

class Dataset_Generator(Dataset):
    def __init__(self, brains, labels, masks, transform = None):
        self.brains = brains.astype(float)
        self.labels = labels
        self.masks = masks
        self.brain_idx = 0
        self.end_brain_idx = 0
        self.patches_data = np.zeros((6,0)+PATCH_SIZE)
        self.patches_label = np.zeros((1,0)+PATCH_SIZE)
        self.patches_mask = np.zeros((1,0)+PATCH_SIZE)
        self.transform = transform
    def __len__(self):
        if (self.brains.shape[0] == SPLIT):
            return 169*SPLIT
        return 169*VAL
    def __getitem__(self, idx):
        if (idx == 0):
            self.end_brain_idx = 0
            self.brain_idx = 0
            #  When restarting epoch, mix up the order of the brains (only for training)
            if self.transform is not None:
                perm = np.random.permutation(len(self.brains))
                self.brains = self.brains[perm]
                self.labels = self.labels[perm]
                self.masks = self.masks[perm]
        if (idx == self.end_brain_idx):
            #  If you have reached last index of the extracted brain, extract a new brain
            patches_generating = True
            while (patches_generating):
                if (idx != 0):
                    self.brain_idx+=1
                #  extract t2 signal patches
                self.patches_data = generatePatches(self.brains[self.brain_idx], self.masks[self.brain_idx])
                self.end_brain_idx += self.patches_data.shape[0]
                if (self.patches_data.shape[0] != 0):
                    patches_generating = False
                #  extract label patches
                self.patches_label = generatePatches(self.labels[self.brain_idx], self.masks[self.brain_idx])
                self.patches_mask = generatePatches(self.masks[self.brain_idx], self.masks[self.brain_idx])
                #  if training, shuffle the patches
                if self.transform is not None:
                    perm = np.random.permutation(len(self.patches_data))
                    self.patches_data = self.patches_data[perm]
                    self.patches_label = self.patches_label[perm]
                    self.patches_mask = self.patches_mask[perm]
        #  normalize the signal curve
        signal_graph = self.patches_data[idx - self.end_brain_idx]/(self.patches_data[idx - self.end_brain_idx][0] + 1e-16)
        signal_label = self.patches_label[idx - self.end_brain_idx]
        patches_mask = self.patches_mask[idx - self.end_brain_idx]
        if self.transform is not None:
            idx = random.randrange(0,len(self.transform))
            subject = tio.Subject(one_image=tio.ScalarImage(tensor=signal_graph))
            signal_graph = self.transform[idx](subject)
            tfm = signal_graph.get_composed_history()
            signal_graph = signal_graph['one_image'].data
            subject = tio.Subject(one_image=tio.ScalarImage(tensor=signal_label))
            signal_label = tfm(subject)
            signal_label = signal_label['one_image'].data
            subject = tio.Subject(one_image=tio.ScalarImage(tensor=patches_mask[None]))
            patches_mask = tfm(subject)
            patches_mask = patches_mask['one_image'].data
            return (signal_graph.float() * patches_mask.float(), signal_label.float() * patches_mask.float())
        return (torch.tensor(signal_graph.copy()).float(), torch.tensor(signal_label.copy()).float())

def dataset_and_dataloader_creator(data, label, mask, transform = None):
    DS = Dataset_Generator(data, label, mask, transform)
    DL = DataLoader(DS, batch_size=7, shuffle=False) #if is_deconv, bs = 8
    return DS,DL

flip = tio.RandomFlip(axes=('LR'))
rotateAxial = tio.RandomAffine(scales=(1,1), degrees=(0,0,0,0,-5,5))
rotateAndFlip = tio.Compose([rotateAxial, flip])
noise = tio.RandomNoise(std=(0.02))
transforms = [flip, rotateAndFlip, noise  ]

TRAIN_DS, TRAIN_DL = dataset_and_dataloader_creator(trainBrains, trainLabels, trainMaskArray, transforms)
VALID_DS, VALID_DL = dataset_and_dataloader_creator(validBrains, validLabels, validMaskArray)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning.logging import TensorBoardLogger
from Hang.unet3dPersonalGroupNorm_pytorchLightning import unet3d

CHECKPOINT = False

weight = 0
model = unet3d(0.001, decay_factor = 0.2, n_classes = 1).float()

filepath = f'../unet_models/full_brain_{weight}'
checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    save_top_k=60,
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

loggerName = f'50_brains_{weight}'
logger = TensorBoardLogger("lightning_logs_50_gpu0", name=loggerName)

if (CHECKPOINT):
    trainer = Trainer(max_epochs=300, gpus=[2], logger=logger,
                      checkpoint_callback=checkpoint_callback, 
                      early_stop_callback=early_stop_callback, 
                      accumulate_grad_batches=2, 
                      auto_lr_find=False, 
                      resume_from_checkpoint="../unetpp_big_gpu0/_ckpt_epoch_11.ckpt")
else:
    trainer = Trainer(max_epochs=300, gpus=[2], logger=logger, 
                      accumulate_grad_batches=2, 
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback, 
                      auto_lr_find=False)

trainer.fit(model, TRAIN_DL, VALID_DL)
