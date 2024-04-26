import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import anndata as ad
import torch
import hdf5plugin
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm
from torch import nn

device=0

data = ad.read_h5ad('Lung_merged.h5ad')
imgs = torch.load('Lung_imgs_merged.pt')
# for d in imgs:
#     for i in imgs[d]:
#         imgs[d][i] = torch.from_numpy(imgs[d][i])
        
def corr(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c, 1) / torch.sqrt(torch.sum(y_true_c * y_true_c, 1)+1e-8) / torch.sqrt(
        torch.sum(y_pred_c * y_pred_c, 1)+1e-8))
    return pearson

class CustomImageDataset(Dataset):
    def __init__(self, data, imgs, id_list, r=112):
        self.imgs = imgs
        self.data = torch.from_numpy(data.X)
        self.mean = torch.tensor([19.528536058021583, 19.832453526270477, 40.23390620591592])[:, None, None]
        self.std = torch.tensor([18.334882013677042, 17.656087119286127, 32.736667837948936])[:, None, None]
        self.r = r
        self.datasets = data.obs['dataset'].values
        self.fovs = data.obs['fov'].values
        self.xs = data.obs['CenterY_local_px'].values
        self.ys = data.obs['CenterX_local_px'].values
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        whole_image = self.imgs[self.datasets[idx]][self.fovs[idx]]
        x = 3647-self.xs[idx]
        y = self.ys[idx]
#         print(x, y)
        image = whole_image[x-self.r: x+self.r, y-self.r:y+self.r]
        image = image.permute(2,0,1)
        image = (image.float() - self.mean)/self.std
        label = self.data[idx]
        return image, label

random.seed(0)
np.random.seed(0)
data.obs['batch'] = data.obs['dataset'].astype('str') + '_' + data.obs['fov'].astype('str')
batch = list(data.obs['batch'].unique())
random.shuffle(batch)

res = []
for i in range(5):
    test_bch = batch[int(len(batch)/5*i): int(len(batch)/5*(i+1))]
    train_bch = batch[:int(len(batch)/5*i)] + batch[int(len(batch)/5*(i+1)):]
    train = CustomImageDataset(data, imgs, np.arange(data.shape[0])[data.obs['batch'].isin(train_bch)])
    train_loader = DataLoader(train, batch_size=192, shuffle=True, sampler=None,
               batch_sampler=None, num_workers=64, drop_last=True, pin_memory=True)
    val = CustomImageDataset(data, imgs, np.arange(data.shape[0])[data.obs['batch'].isin(test_bch)])
    val_loader = DataLoader(val, batch_size=256, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=64, drop_last=False, pin_memory=True)
    
#     model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl').to(device)
    
#     from vits import VisionTransformerMoCo
#     model = VisionTransformerMoCo(pretext_token=True, global_pool='avg')
#     model.head = nn.Linear(768, 980)
#     checkpoint = torch.load('./PathoDuet/checkpoint_IHC.pth', map_location="cpu")
#     model.load_state_dict(checkpoint, strict=False)
#     model.to(device)

#     model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=False).to(device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True).to(device)
    
    mse = MSELoss()
    optim = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    vals = []
    for epoch in range(100):
        losses = []
        for i, (x, y) in tqdm(enumerate(train_loader)):
            model.train()
            pred = F.relu(model(x.to(device))[:, :980])#[1])#
            loss = mse(pred, y.to(device)) - corr(pred, y.to(device)) * 10
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss = sum(losses)/len(losses)
        print('Start evaluation.')
        
        with torch.no_grad():
            model.eval()
            losses = []
            for _, (x, y) in tqdm(enumerate(val_loader)):
                pred = model(x.to(device))[:, :980]#[1]#
                losses.append(corr(F.relu(pred).float(), y.to(device)).item())
            vals.append(sum(losses)/len(losses))
            if max(vals) != max(vals[-5:]):
                break
        print('Train:', loss, 'Val:', sum(losses)/len(losses))
        losses = []
    print('Fold', i, 'Final val:', max(vals))
    res.append(max(vals))
print(res)
