
from PIL import Image
import sys
from sklearn.decomposition import NMF
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import pyod as od
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
device = 'cuda'
import torch.nn as nn
from pyod.models.copod import COPOD
from pyod.models.iforest  import IForest
import pickle
path = os.getcwd()

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = torch.flatten(image)
        sample = {'img': image,
                  'shape': image.shape,
                  'label': int(self.img_list[idx][1])}
        return sample


    def img(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = torch.flatten(image)
        return image

    def label(self, idx):
        return int(self.img_list[idx][1])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

trainset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/datasplit/trainCT_COVID.txt',
      txt_NonCOVID=path+'/datasplit/trainCT_NonCOVID.txt',
      transform= train_transformer)
valset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/datasplit/valCT_COVID.txt',
      txt_NonCOVID=path+'/datasplit/valCT_NonCOVID.txt',
      transform= val_transformer)
testset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/datasplit/testCT_COVID.txt',
      txt_NonCOVID=path+'/datasplit/testCT_NonCOVID.txt',
      transform= val_transformer)
# print(trainset.__len__())
# print(valset.__len__())
# print(testset.__len__())

train_x = np.zeros((425,150528))
train_y = np.zeros((425,1))
for i in range(trainset.__len__()):
    train_x[i] = trainset.img(i)
    train_y[i] = trainset.label(i)

val_x = np.zeros((118,150528))
val_y = np.zeros((118,1))
for i in range(valset.__len__()):
    val_x[i] = valset.img(i)
    val_y[i] = valset.label(i)

test_x = np.zeros((203, 150528))
test_y = np.zeros((203,1))
for i in range(testset.__len__()):
    test_x[i] = testset.img(i)
    test_y[i] = testset.label(i)