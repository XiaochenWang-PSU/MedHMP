#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np
import model 
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sparse
from tqdm import tqdm
import pickle



MODEL_PATH = "model/"
device = torch.device("cuda:0")
EPOCHS = 200

BiAE = model.Bimodal_AE(seq_len = 48, n_features = 1318, ts_embedding_dim = 256, tb_embedding_dim = 256).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(BiAE.parameters(),
                              lr = 5e-4,
                              weight_decay = 1e-8) 


data = pickle.load(open("data/stay_pretrain_data.p", 'rb'))

for epoch in tqdm(range(EPOCHS)):
    
    loss = 0
 
    
    data__ = DataLoader(data, batch_size = 128, shuffle = True)
    for batch_idx, batch_data in enumerate(data__):
        

        X = batch_data[0].to(torch.float32).to(device)
        S = batch_data[1].to(torch.float32).to(device)
        optimizer.zero_grad()
        outputs = BiAE(X,S).to(device)
        train_loss = criterion(outputs, X)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    print("Loss = ", loss)
    torch.save(BiAE,MODEL_PATH + 'stay_pretrained.p')
    
