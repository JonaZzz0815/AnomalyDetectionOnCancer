'''
Author: your name
Date: 2022-01-01 15:49:11
LastEditTime: 2022-01-09 08:33:26
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /AnomalyDetectionOnCancer/MAE/utils.py
'''
import random
import torch
import numpy as np
from sklearn.metrics import f1_score, auc 
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def measure(predicts,labels):   
    acc = (predicts==labels).sum().float() / len(predicts)
    f1 = f1_score(labels,predicts)
    auc_score = auc(labels,predicts)
    return acc,f1,auc_score


