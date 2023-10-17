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
import pandas as pd
import pickle
import ast
import copy
from pytorch_metric_learning import losses
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from info_nce import InfoNCE


MODEL_PATH = "model/"




device = torch.device("cuda:0")

data = pickle.load(open("data/admission_pretrain_data.p", 'rb'))


def collate_batch(batch_data):



        icd = torch.tensor([i[0] for i in batch_data]).to(torch.float32).to(device)
        drug = torch.tensor([i[1] for i in batch_data]).to(torch.float32).to(device)
        X = torch.tensor(np.array([np.stack(i[2], axis = 0) for i in batch_data])).to(torch.float32).to(device)
        S = torch.tensor(np.array([np.stack(i[3], axis = 0) for i in batch_data])).to(torch.float32).to(device)
        text = torch.stack([i[4] for i in batch_data]).to(torch.float32).to(device) 
        return [icd, drug, X, S, text]
data__ = DataLoader(data, batch_size = 4096, shuffle = True, collate_fn=collate_batch)

EPOCHS = 300

HADM_AE = model.HADM_AE(vocab_size1 = 7686+1, vocab_size2 = 1701+1,  d_model = 256, dropout=0.1, dropout_emb=0.1, length=48).to(device)
criterion_mcp = torch.nn.MSELoss()
criterion_cl = InfoNCE()
optimizer = torch.optim.Adam(HADM_AE.parameters(),
                              lr = 2e-5,
                              weight_decay = 1e-8)
enc = torch.load(MODEL_PATH + "/stay_pretrained.p").state_dict()
model_dict = HADM_AE.state_dict()

state_dict = {k.replace("encoder.", "enc.ICU_Encoder."):v for k,v in enc.items() if k.replace("encoder.", "enc.ICU_Encoder.") in model_dict.keys()}
print(state_dict.keys())

HADM_AE.state_dict().update(state_dict)

HADM_AE.load_state_dict(state_dict, strict=False)



step = 0

for epoch in tqdm(range(EPOCHS)):
    
    loss = 0
    
    
    for batch_idx, batch_data in enumerate(data__):
        
        print(step)

        icd = batch_data[0]
        drug = batch_data[1]
        X = batch_data[2]
        S = batch_data[3]
        text = batch_data[4]
        mask_icd = (torch.rand(size=(icd.shape)) > 0.15).to(device)
        masked_icd = ~mask_icd*icd
        nums_masked_icd = (masked_icd).sum(dim=1).unsqueeze(1)
        unmasked_icd = icd*mask_icd
        mask_icd[:, -1] = 1
        
        
        mask_drug = (torch.rand(size=(drug.shape)) > 0.15).to(device)
        masked_drug = ~mask_drug*drug
        nums_masked_drug = (~mask_drug*drug).sum(dim=1).unsqueeze(1)
        unmasked_drug = drug*mask_drug
        masked_drug[:,-1] = 1
        optimizer.zero_grad()
            


        doc_emb, doc_rep, x1, x1_rep, x2, x2_rep, mcm_x1_rep, mcm_x2_rep = HADM_AE(icd,drug,nums_masked_icd, nums_masked_drug,  X,S, text, unmasked_icd,unmasked_drug)
        mcm_loss = (criterion_mcp(mcm_x1_rep, masked_icd) + criterion_mcp(mcm_x2_rep, masked_drug))/2
        cl_loss =  (criterion_cl(doc_emb, doc_rep) + criterion_cl(x1, x1_rep) + criterion_cl(x2, x2_rep))/3
 
        train_loss = mcm_loss + 0.1*cl_loss
        train_loss.backward()

        optimizer.step()

        loss += train_loss.item()
        
        step+=1
    print("one step mcm:", mcm_loss)
    print("one step cl:", cl_loss)
    print("Loss = ", loss)

    torch.save(HADM_AE,MODEL_PATH + 'admission_pretrained.p')