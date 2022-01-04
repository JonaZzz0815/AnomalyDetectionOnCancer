
from PIL import Image
import sys
# from SimpleITK.SimpleITK import ImageReaderBase_GetImageIOFromFileName
from sklearn.decomposition import NMF
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import pyod as od
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
import cv2
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
device = 'cuda'
import torch.nn as nn
from pyod.models.copod import COPOD
from pyod.models.iforest  import IForest
import pickle
# from lungmask import mask
# import SimpleITK as sitk
# import pydicom
# import radiomics


path = str(os.getcwd())+"/COVID-CT"

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data
#Prepocess #
def normalize_image(x): #normalize image pixels between 0 and 1
        if np.max(x)-np.min(x) == 0 and np.max(x) == 0:
            return x
        elif np.max(x)-np.min(x) == 0 and np.max(x) != 0:
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))



def extract(img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        sift =  cv2.SIFT_create()
        step_size = 16
        kp = [cv2.KeyPoint(x,y,step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        des =  sift.compute(img,kp)
        return torch.flatten(torch.from_numpy(des[1])) 

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
        # image = sitk.ReadImage(img_path)
        # image = mask.apply(image)
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
        # image = sitk.ReadImage(img_path,imageIO="PNGImageIO")
        # image = mask.apply(image)
        # image = cv2.imread(img_path,cv2.COLOR_BGR2GRAY)
        # image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        image = extract(image)
        return image

    def label(self, idx):
        return int(self.img_list[idx][1])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0))
    #transforms.RandomHorizontalFlip(),
    # transforms.ToTensor()
    # normalize
])
val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)
    # transforms.ToTensor(),
    # normalize
])



# print(trainset.__len__())
# print(valset.__len__())
# print(testset.__len__())
num_feature = 25088
def GetTrainSet(path):
    trainset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/COVID-CT/datasplit/COVID/trainCT_COVID.txt',
      txt_NonCOVID=path+'/COVID-CT/datasplit/NonCOVID/trainCT_NonCOVID.txt',
      transform= train_transformer)
    train_x = np.zeros((425,num_feature))
    train_y = np.zeros((425,1))
    for i in range(trainset.__len__()):
        train_x[i] = trainset.img(i)
        train_y[i] = trainset.label(i)
    return train_x,train_y

def GetValSet(path):
    valset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/COVID-CT/datasplit/COVID/valCT_COVID.txt',
      txt_NonCOVID=path+'/COVID-CT/datasplit/NonCOVID/valCT_NonCOVID.txt',
      transform= val_transformer)

    val_x = np.zeros((118,num_feature))
    val_y = np.zeros((118,1))
    for i in range(valset.__len__()):
        val_x[i] = valset.img(i)
        val_y[i] = valset.label(i)
    return val_x,val_y


def GetTestSet(path):
    testset = CovidCTDataset(root_dir=path,
      txt_COVID=path+'/COVID-CT/datasplit/COVID/testCT_COVID.txt',
      txt_NonCOVID=path+'/COVID-CT/datasplit/NonCOVID/testCT_NonCOVID.txt',
      transform= val_transformer)
    test_x = np.zeros((203, num_feature))
    test_y = np.zeros((203,1))
    for i in range(testset.__len__()):
        test_x[i] = testset.img(i)
        test_y[i] = testset.label(i)
    return test_x,test_y