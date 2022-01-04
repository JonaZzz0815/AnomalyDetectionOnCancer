'''
Author: your name
Date: 2022-01-03 15:38:00
LastEditTime: 2022-01-03 20:07:56
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /AnomalyDetectionOnCancer/MAE/train.py
'''
import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from Data import *
from model import *
from utils import setup_seed
def CalAcc(losses,labels):
    lb = min(losses)
    hb = max(losses)
    max_acc = 0
    for t in torch.arange(lb,hb,0.001):
        given_label = losses < t
        acc = (given_label == labels).sum() / float(len (labels))
        if (acc > max_acc):
            max_acc = acc
    
    return max_acc;
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_device_batch_size', type=int, default=8)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch.pt')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    path = os.path.abspath(os.path.dirname(os.getcwd()))
    train_dataset = GetTrainSet(path)
    test_dataset = GetTestSet(path)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        writer = SummaryWriter(os.path.join('logs', 'COVID-CT', 'pretrain-cls'))
    else:
        model = MAE_ViT()
        writer = SummaryWriter(os.path.join('logs', 'COVID-CT', 'scratch-cls'))
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    
        
    with torch.no_grad():
        losses = []
        acces = []
        labels=np.asarray([test_dataset[i][1] for i in range(len(test_dataset))])
        labels = torch.tensor(labels)
        test_img = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        print(len(test_img))
        
        img = test_img.to(device)
        predicted_img, mask = model(img)
        predicted_img = predicted_img * mask + img * (1 - mask)
        loss1 = (predicted_img - img) ** 2 * mask / 0.75
        loss = torch.tensor([torch.mean(item) for item in loss1])
        print(loss)
        acc = CalAcc(loss,labels)
        print(acc)
        
        