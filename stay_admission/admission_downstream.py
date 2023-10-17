#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json 
import pickle
import pandas as pd
import numpy as np
import sparse
import torch
import model
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sklearn
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
from baseline import *
from operations import *
from train import *
from model import *
import sys
import csv
import os






print(sys.argv[2:])
EPOCHS,LR, BATCH, SEED, TASK, DEVICE = sys.argv[2:]
EPOCHS,LR, BATCH, SEED, TASK, DEVICE = int(EPOCHS), float(LR), int(BATCH), int(SEED), str(TASK), str(DEVICE)

PATH = "hmp_results/" + TASK + '/'

if str(EPOCHS) + '_' + str(LR) + '_' + str(BATCH)+ '_' +'.csv' in os.listdir(PATH):
    print("conducted experiments")
else:


    device = torch.device(DEVICE)

    data = pickle.load(open("data/hmp_admission.p", 'rb'))
    

    split_mark = int(len(data)*0.8), int(len(data)*0.9)
    
    

    def collate_batch(batch_data):



            icd = torch.tensor([i[0] for i in batch_data]).to(torch.float32).to(device)
            drug = torch.tensor([i[1] for i in batch_data]).to(torch.float32).to(device)
            X = torch.tensor(np.array([np.stack(i[2], axis = 0) for i in batch_data])).to(torch.float32).to(device)
            S = torch.tensor(np.array([np.stack(i[3], axis = 0) for i in batch_data])).to(torch.float32).to(device)
            input_ids =  torch.stack([i[4] for i in batch_data]).to(device)
            attention_mask = torch.stack([i[5] for i in batch_data]).to(device)
            token_type_ids = torch.stack([i[6] for i in batch_data]).to(device)
            label = torch.tensor(np.array([i[-1] for i in batch_data])).to(torch.float32).to(device) 
            return [icd , drug, X, S, input_ids, attention_mask, token_type_ids, label]
    def collate_batch_ts(batch_data):



            X = torch.tensor(np.array([i[0] for i in batch_data])).to(torch.float32).to(device)
            label = torch.tensor(np.array([i[4] for i in batch_data])).to(torch.float32).to(device) 
            return [X, label]
        

    
    
    
    
    
    
    
    
    # multimodal encoder evaluation
    
    test = DataLoader(data[split_mark[1]:], batch_size = BATCH, shuffle = True, collate_fn=collate_batch)
    train = DataLoader(data[:split_mark[0]], batch_size = BATCH, shuffle = True, collate_fn=collate_batch)
    valid = DataLoader(data[split_mark[0]:split_mark[1]], batch_size = BATCH, shuffle = True, collate_fn=collate_batch)
    
    
    
    MedHMP = HADM_CLS(7686+1, 1701+1, 256, 0.8).to(device)
    MedHMP.train()
    enc = torch.load("model/admission_pretrained.p").state_dict()
    model_dict = MedHMP.state_dict()
    state_dict = {k.split('enc.')[-1]:v for k,v in enc.items() if k.split('enc.')[-1] in model_dict.keys()}
    MedHMP.state_dict().update(state_dict)
    MedHMP.load_state_dict(state_dict, strict = False)
    MedHMP, _ = adm_trainer(MedHMP, train, valid, test,EPOCHS, LR, BATCH, SEED, device , encoder = 'HMP', patience = 5)
    file = open(PATH +str(EPOCHS) + '_' + str(LR) + '_' + str(BATCH)+ '_' +'.csv','w',encoding  = 'gbk')
    csv_w = csv.writer(file)
    metrics = list(eval_metric_admission(test, MedHMP,  device, 'HMP'))[:-1]
    csv_w.writerow(["MedHMP"] + metrics)
    file.close()
    
    globals().clear()



















