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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score,precision_recall_curve, auc
import matplotlib.pyplot as plt
from baseline import *
from operations import *
from focal_loss.focal_loss import FocalLoss
import os




def eval_metric_stay(eval_set, model,  device, encoder = 'normal'):
    
    model.eval()
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, batch_data in enumerate(eval_set):
            X = batch_data[0]                                                                      
            if encoder == 'HMP':                                                                   
                S = batch_data[1]                                                                                                                
            elif encoder == 'BERT':                                                                
                S = batch_data[1]                                                                  
                input_ids = batch_data[2]                                                          
                attention_mask = batch_data[3]                                                     
                token_type_ids = batch_data[4]                                                     
            labels = batch_data[-1]                                                                
            if encoder == 'normal':
                X2 = batch_data[1]           
                S = batch_data[2]                                                          
                # print(X.shape)                                                                   
                outputs = model(X, X2, S).squeeze().to(device)                                            
            elif encoder == 'HMP':                                                                 
                outputs = model(X,S).squeeze().to(device)                
            elif encoder == 'BERT':                                                                
                outputs = model(X,S,input_ids, token_type_ids, attention_mask).squeeze().to(device)
            score = outputs
            score = score.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            pred = np.where(score >= 0.5, 1.0, 0.0)

            if labels.shape[0] != 1:
                
                y_true = np.concatenate((y_true, labels))
                y_pred = np.concatenate((y_pred, pred))
                y_score = np.concatenate((y_score, score))
            else:
                y_true = np.array(list(y_true) + list(labels))
                y_pred = np.array(list(y_pred) + list([pred]))
                y_score = np.array(list(y_score) + list([score]))
        accuary = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        kappa = cohen_kappa_score(y_true, y_pred)
        loss = criterion(torch.from_numpy(y_true), torch.from_numpy(y_score))

    return  f1, roc_auc, pr_auc, kappa, loss

def eval_metric_admission(eval_set, model,  device, encoder = 'normal'):
    
    model.eval()
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, batch_data in enumerate(eval_set):
            icd = batch_data[0]
            drug = batch_data[1]
            X = batch_data[2]
            S = batch_data[3]
            input_ids = batch_data[4]
            attention_mask = batch_data[5]
            token_type_ids = batch_data[6]
            labels = batch_data[-1]
            outputs = model(icd, drug,X,S,input_ids, attention_mask, token_type_ids).squeeze().to(device)
            score = outputs
            score = score.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
                # pred = torch.tensor([1 if x > 0.5 else 0 for x in score])
    
            pred = np.where(score >= 0.5, 1.0, 0.0)
            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, pred))
            y_score = np.concatenate((y_score, score))
        accuary = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        kappa = cohen_kappa_score(y_true, y_pred)
        loss = criterion(torch.from_numpy(y_true), torch.from_numpy(y_score))

    return  f1, roc_auc, pr_auc, kappa, loss


def icu_trainer(model, train, valid, test, epoch, learn_rate, batch_size, seed, device, encoder = 'normal', patience = 3):
    
    torch.manual_seed(seed)
    
    model.train()
    aupr_list = []

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                momentum=0.9,
                                  lr = learn_rate,
                                  weight_decay = 1e-2)
    
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
    f1, roc_auc, pr_auc, kappa, valid_loss = eval_metric_stay(valid, model, device, encoder)
    best_dev = valid_loss
    best_epoc = 0
    model.train()
    from datetime import datetime
    dt = datetime.now()
    torch.save(model, "saved_model/hmp_model" + str(dt) +".p")
    best_name = "saved_model/hmp_model" + str(dt) +".p"

    for epoch in tqdm(range(epoch)):
        
        loss = 0
        model.train()
        for batch_idx, batch_data in enumerate(train):
            
                X = batch_data[0]
                if encoder == 'HMP':
                    S = batch_data[1]
                elif encoder == 'BERT':
                    S = batch_data[1]
                    input_ids = batch_data[2]
                    attention_mask = batch_data[3]
                    token_type_ids = batch_data[4]
                label = batch_data[-1]
                optimizer.zero_grad()
                if encoder == 'normal':
                    outputs = model(X).squeeze().to(device)
                elif encoder == 'HMP':
                    outputs = model(X,S).squeeze().to(device)
                elif encoder == 'BERT':
                    outputs = model(X,S,input_ids, attention_mask, token_type_ids).squeeze().to(device)
                train_loss = criterion(outputs, label)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
        scheduler.step()
        print("Training Loss = ", loss)
        
        
    
        model.eval()
        

        f1, roc_auc, pr_auc, kappa, valid_loss = eval_metric_stay(valid, model, device, encoder)
        print("Dev = ", valid_loss)
        
        dt = datetime.now()
        if best_dev > valid_loss:
            best_dev = valid_loss
            best_epoc = epoch
            torch.save(model, "saved_model/icu_model" + str(dt) +".p")
            os.remove(best_name)
            best_name = "saved_model/icu_model" + str(dt) +".p"
        if epoch - best_epoc == patience:
            break
    model.train()
    model = torch.load(best_name)
    os.remove(best_name)
    return model, aupr_list



def adm_trainer(model, train, valid, test, epoch, learn_rate, batch_size, seed, device, encoder = 'normal', patience = 3):
    
    torch.manual_seed(seed)
    
    model.train()
    aupr_list = []

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                momentum=0.9,
                                  lr = learn_rate,
                                  weight_decay = 1e-2)
    
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
    f1, roc_auc, pr_auc, kappa, valid_loss = eval_metric_admission(valid, model, device, encoder)
    best_dev = valid_loss
    best_epoc = 0
    model.train()
    
    for epoch in tqdm(range(epoch)):
        
        loss = 0
        
        for batch_idx, batch_data in enumerate(train):
                icd = batch_data[0]
                drug = batch_data[1]
                X = batch_data[2]
                S = batch_data[3]
                input_ids = batch_data[4]
                attention_mask = batch_data[5]
                token_type_ids = batch_data[6]
                label = batch_data[-1]
                optimizer.zero_grad()
                outputs = model(icd, drug,X,S,input_ids, attention_mask, token_type_ids).squeeze().to(device)
                train_loss = criterion(outputs, label)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
        scheduler.step()
        print("Training Loss = ", loss)
        
        
    
        model.eval()
        

        f1, roc_auc, pr_auc, kappa, valid_loss = eval_metric_admission(valid, model, device, encoder)
        print("Dev = ", valid_loss)
        if best_dev > valid_loss:
            best_dev = valid_loss
            best_epoc = epoch
            torch.save(model, "saved_model/adm_model.p")
        if epoch - best_epoc == patience:
            model = torch.load("saved_model/adm_model.p")
            break
        model.train()
    model = torch.load("saved_model/adm_model.p")
    return model, aupr_list



   

    
   
