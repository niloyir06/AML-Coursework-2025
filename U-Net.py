import os
import json
import csv
import random
import pickle
import time
import math
import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from sklearn.model_selection import train_test_split

start = time.time()

def args_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--total_epoch", type=int, default=10)
    parser.add_argument("--ORIGA_path", type=str, help="path to ORIGA Data")
    parser.add_argument("--G1020_path", type=str, help="path to G1020 Data")
    parser.add_argument("--REFUGE_path", type=str, help="path to REFUGE Data")

    args = parser.parse_args()
    return args

### Dataset

class GlaucomaDataset(Dataset):

    def __init__(self, root_dir, split='train', output_size=(256,256)):
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        self.images = []
        self.segs = []
        # Load data index
        for direct in self.root_dir:
            self.image_filenames = []
            for path in os.listdir(os.path.join(direct, "Images_Square")):
                if(not path.startswith('.')):
                    self.image_filenames.append(path)


            for k in range(len(self.image_filenames)):
                print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                img_name = os.path.join(direct, "Images_Square", self.image_filenames[k])
                #img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
                img = np.array(Image.open(img_name).convert('RGB'))
                img = transforms.functional.to_tensor(img)
                img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
                self.images.append(img)
            if split != 'test':
                for k in range(len(self.image_filenames)):
                    print('Loading {} segmentation {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                    seg_name = os.path.join(direct, "Masks_Square", self.image_filenames[k][:-3] + "png")
                    mask = np.array(Image.open(seg_name, mode='r'))
                    od = (mask==1.).astype(np.float32)
                    oc = (mask==2.).astype(np.float32)
                    od = torch.from_numpy(od[None,:,:])
                    oc = torch.from_numpy(oc[None,:,:])
                    od = transforms.functional.resize(od, output_size, interpolation=Image.NEAREST)
                    oc = transforms.functional.resize(oc, output_size, interpolation=Image.NEAREST)
                    self.segs.append(torch.cat([od, oc], dim=0))

            print('Succesfully loaded {} dataset.'.format(split) + ' '*50)
            
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.split == 'test':
            return img
        else:
            seg = self.segs[idx]
            return img, seg
            
### Preprocessing

def remove_nerves(image):
    img = array_to_img(image)
    
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    # convert image to grayScale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
   
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
  
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    # inpaint with original image and threshold image
    final_image = cv2.inpaint(img,threshold,1,cv2.INPAINT_TELEA)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    
    return final_image.astype(np.float64)/255.0
    
### Metrics


def compute_dice_coef(input, target):
    '''
    Compute dice score metric.
    '''
    batch_size = input.shape[0]
    return sum([dice_coef_sample(input[k,:,:], target[k,:,:]) for k in range(batch_size)])/batch_size

def dice_coef_sample(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2. * intersection) / (iflat.sum() + tflat.sum())


def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''

    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=1)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter, axis=1)

    # return it
    return diameter



def vertical_cup_to_disc_ratio(od, oc):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    '''
    # compute the cup diameter
    cup_diameter = vertical_diameter(oc)
    # compute the disc diameter
    disc_diameter = vertical_diameter(od)
    EPS = 1e-7

    return cup_diameter / (disc_diameter + EPS)

def compute_vCDR_error(pred_od, pred_oc, gt_od, gt_oc):
    '''
    Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
    '''
    pred_vCDR = vertical_cup_to_disc_ratio(pred_od, pred_oc)
    gt_vCDR = vertical_cup_to_disc_ratio(gt_od, gt_oc)
    vCDR_err = np.mean(np.abs(gt_vCDR - pred_vCDR))
    return vCDR_err, pred_vCDR, gt_vCDR


def classif_eval(classif_preds, classif_gts):
    '''
    Compute AUC classification score.
    '''
    auc = roc_auc_score(classif_gts, classif_preds)
    return auc
    
### Post Processing

def refine_seg(pred):
    '''
    Only retain the biggest connected component of a segmentation map.
    '''
    np_pred = pred.numpy()
        
    largest_ccs = []
    for i in range(np_pred.shape[0]):
        labeled, ncomponents = label(np_pred[i,:,:])
        bincounts = np.bincount(labeled.flat)[1:]
        if len(bincounts) == 0:
            largest_cc = labeled == 0
        else:
            largest_cc = labeled == np.argmax(bincounts)+1
        largest_cc = torch.tensor(largest_cc, dtype=torch.float32)
        largest_ccs.append(largest_cc)
    largest_ccs = torch.stack(largest_ccs)
    
    return largest_ccs     
    
### Network

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epoch = 0

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        factor = 2 
        self.down6 = Down(2048, 4096 // factor)
        self.up1 = Up(4096, 2048 // factor)
        self.up2 = Up(2048, 1024 // factor)
        self.up3 = Up(1024, 512 // factor)
        self.up4 = Up(512, 256 // factor)
        self.up5 = Up(256, 128 // factor)
        self.up6 = Up(128, 64)
        self.output_layer = OutConv(64, n_classes)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        out = self.up1(x7, x6)
        out = self.up2(out, x5)
        out = self.up3(out, x4)
        out = self.up4(out, x3)
        out = self.up5(out, x2)
        out = self.up6(out, x1)
        out = self.output_layer(out)
        out = torch.sigmoid(out)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            
        )
        

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
            
        )
        

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        


class OutConv(nn.Module):
    '''
    Simple convolution.
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        

    def forward(self, x):
        return self.conv(x)
        
### Input Parameters

args = args_input()

root_dirs = [args.ORIGA_path,args.G1020_path]
val_dir = [args.REFUGE_path]
lr = args.lr
batch_size = 8
num_workers = 8
total_epoch = args.total_epoch 

### Load Data

train_set = GlaucomaDataset(root_dirs, 
                          split='train')

val_set = GlaucomaDataset(val_dir, 
                        split='val')

train_loader = DataLoader(train_set, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers,
                          pin_memory=True,
                         )
val_loader = DataLoader(val_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True,
                        )
                        
### Init Model

# Device
device = torch.device("cuda:0")

# Network
model = UNet(n_channels=3, n_classes=2).to(device)

# Loss
seg_loss = torch.nn.BCELoss(reduction='mean')

# Optimizer

opt = args.opt

if opt=="sgd":
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
else:  
  optimizer = optim.Adam(model.parameters(), lr=lr)


### Train

# Define parameters
nb_train_batches = len(train_loader)
nb_val_batches = len(val_loader)
nb_iter = 0
best_val_auc = 0.
iters = list(range(1, 10))
val_losses = []
train_losses = []
train_accuracy=[]
val_accuracy=[]


while model.epoch < total_epoch:
    # Accumulators
    train_vCDRs, val_vCDRs = [], []
    train_loss, val_loss = 0., 0.
    train_dsc_od, val_dsc_od = 0., 0.
    train_dsc_oc, val_dsc_oc = 0., 0.
    train_vCDR_error, val_vCDR_error = 0., 0.
    
    ############
    # TRAINING #
    ############
    model.train()
    train_data = iter(train_loader)
    for k in range(nb_train_batches):
        # Loads data
        imgs, seg_gts = next(train_data)
        imgs, seg_gts = imgs.to(device), seg_gts.to(device)

        # Forward pass
        logits = model(imgs)
        loss = seg_loss(logits, seg_gts)
 
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / nb_train_batches
        #for printing the loss curves
        
        train_losses.append(train_loss)
        
        with torch.no_grad():
            # Compute segmentation metric
            pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            #pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu())
            pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            #pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu())
            gt_od = seg_gts[:,0,:,:].type(torch.int8)
            gt_oc = seg_gts[:,1,:,:].type(torch.int8)
            dsc_od = compute_dice_coef(pred_od, gt_od)
            dsc_oc = compute_dice_coef(pred_oc, gt_oc)
            train_dsc_od += dsc_od.item()/nb_train_batches
            train_dsc_oc += dsc_oc.item()/nb_train_batches


            # Compute and store vCDRs
            vCDR_error, pred_vCDR, gt_vCDR = compute_vCDR_error(pred_od.cpu().numpy(), pred_oc.cpu().numpy(), gt_od.cpu().numpy(), gt_oc.cpu().numpy())
            train_vCDRs += pred_vCDR.tolist()
            train_vCDR_error += vCDR_error  / nb_train_batches
            
        # Increase iterations
        nb_iter += 1
        
        # Std out
        print('Epoch {}, iter {}/{}, loss {:.6f}'.format(model.epoch+1, k+1, nb_train_batches, loss.item()) + ' '*20, 
              end='\r')
    
    ##############
    # VALIDATION #
    ##############
    model.eval()
    with torch.no_grad():
        val_data = iter(val_loader)
        for k in range(nb_val_batches):
            # Loads data
            imgs, seg_gts = next(val_data)
            imgs, seg_gts = imgs.to(device), seg_gts.to(device)
            
            # Forward pass
            logits = model(imgs)
            val_loss += seg_loss(logits, seg_gts).item() / nb_val_batches
            
            val_losses.append(val_loss)

            # Std out
            print('Validation iter {}/{}'.format(k+1, nb_val_batches) + ' '*50, 
                  end='\r')
            
            # Compute segmentation metric
            pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            gt_od = seg_gts[:,0,:,:].type(torch.int8)
            gt_oc = seg_gts[:,1,:,:].type(torch.int8)
            dsc_od = compute_dice_coef(pred_od, gt_od)
            dsc_oc = compute_dice_coef(pred_oc, gt_oc)
            val_dsc_od += dsc_od.item()/nb_val_batches
            val_dsc_oc += dsc_oc.item()/nb_val_batches
            
        
            vCDR_error, pred_vCDR, gt_vCDR = compute_vCDR_error(pred_od.cpu().numpy(), pred_oc.cpu().numpy(), gt_od.cpu().numpy(), gt_oc.cpu().numpy())
            val_vCDRs += pred_vCDR.tolist()
            val_vCDR_error += vCDR_error / nb_val_batches
    print('VALIDATION epoch {}'.format(model.epoch+1)+' '*50)
    print('LOSSES: {:.4f} (train), {:.4f} (val)'.format(train_loss, val_loss))
    print('OD segmentation (Dice Score): {:.4f} (train), {:.4f} (val)'.format(train_dsc_od, val_dsc_od))
    print('OC segmentation (Dice Score): {:.4f} (train), {:.4f} (val)'.format(train_dsc_oc, val_dsc_oc))
    print('vCDR error: {:.4f} (train), {:.4f} (val)'.format(train_vCDR_error, val_vCDR_error))
    # Save model if best validation AUC is reached
    if val_dsc_od + val_dsc_oc > best_val_auc:
        torch.save(model.state_dict(), 'best_seg.pth')
        best_val_auc = val_dsc_od + val_dsc_oc
        print('Best validation AUC reached. Saved model weights.')
    print('_'*50)
        
    # End of epoch
    model.epoch += 1
    
finish = time.time()
min_time = math.floor((finish - start)/60)
sec_time = (finish - start)-min_time*60

print(f"Total time taken: {min_time} minutes and {sec_time:.2f} seconds")
                                      
