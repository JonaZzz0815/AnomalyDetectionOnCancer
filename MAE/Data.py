import os
import numpy as np
import pandas as pd
import itertools
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, utils

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.RandomResizedCrop((256),scale=(0.5,1.0)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
val_transformer = transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize
])

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

seed = 2020
np.random.seed(seed)
# root_dir = os.path.abspath(os.path.dirname(os.getcwd()))
class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None,only = False):
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
        if only == True:
            self.num_cls = 1
        else:# normal case
            self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir+'/COVID-CT', self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
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
        # image = torch.flatten(image)
        # sample = {'img': image,
        #           'shape': image.shape,
        #           'label': int(self.img_list[idx][1])}
        return image, int(self.img_list[idx][1])


def GetTrainSet(path):
    trainset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/COVID-CT/datasplit/COVID/trainCT_COVID.txt',
      txt_NonCOVID=path+'/COVID-CT/datasplit/NonCOVID/trainCT_NonCOVID.txt',
      transform= train_transformer,only = True)
    return trainset

def GetValSet(path,only):
    valset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/COVID-CT/datasplit/COVID/valCT_COVID.txt',
      txt_NonCOVID=path+'/COVID-CT/datasplit/NonCOVID/valCT_NonCOVID.txt',
      transform= val_transformer,only = only)
    return valset


def GetTestSet(path):
    testset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/COVID-CT/datasplit/COVID/testCT_COVID.txt',
      txt_NonCOVID=path+'/COVID-CT/datasplit/NonCOVID/testCT_NonCOVID.txt',
      transform= val_transformer)
    return testset
