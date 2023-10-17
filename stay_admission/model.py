#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch 
from torch import nn, optim
import torch.functional as F
from operations import *
from torchmetrics import AUROC, AveragePrecision
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer




class table_encoder(nn.Module):    
    def __init__(self, input_shape, emb_shape):
        super().__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=73, out_features=64
        )
        self.encoder_output_layer = nn.Linear(
            in_features=64, out_features=emb_shape
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=1, out_features=32
        )
        self.decoder_output_layer = nn.Linear(
            in_features=32, out_features=input_shape
        )
        self.emb_shape = emb_shape
    def forward(self, features):
        bz = features.shape[0]
        activation = self.encoder_hidden_layer(features).squeeze()
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation).reshape(bz, self.emb_shape)
        return code



class LSTM_Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=32):
    super().__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=self.n_features,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=embedding_dim,
      hidden_size=embedding_dim,
      num_layers=2,
      batch_first=True
    )
    self.dropout = nn.Dropout(p = 0.1)
    self.norm = nn.LayerNorm(embedding_dim)
    
  def forward(self, x):
      
    bz = x.shape[0]
    x, (hidden_n, cell) = self.rnn1(x)
    x = self.dropout(x)
    x = self.norm(x)
    x, (hidden_n, cell) = self.rnn2(x)
    return x, cell[-1].reshape(bz, self.embedding_dim)
    
    
class LSTM_Decoder(nn.Module):
  def __init__(self, seq_len, n_features, input_dim=32):
    super().__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = input_dim, n_features
    self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=self.n_features,
      num_layers=1,
      batch_first=True
    )
    self.dropout = nn.Dropout(p = 0.2)
    self.norm = nn.LayerNorm(self.hidden_dim)
    self.output_layer = nn.Linear(self.n_features, self.n_features)
    self.device = torch.device("cuda:1")
    
  def forward(self, x):
    x = torch.unsqueeze(x,1)
    padding = torch.zeros(x.shape).repeat(1, self.seq_len-1, 1).to(self.device)
    x = torch.cat([x,padding], 1)
    x, (hidden_n, cell_n) = self.rnn2(x)
    return x




class Bimodal_AE(nn.Module):
    def __init__(self, seq_len, n_features, ts_embedding_dim=256, tb_embedding_dim=32):
      super().__init__()
      device = torch.device("cuda:1")

      self.decoder = LSTM_Decoder(seq_len, n_features, ts_embedding_dim)
      self.ffn = FFN(ts_embedding_dim)
      self.encoder = ICU_Encoder(seq_len, n_features, ts_embedding_dim = ts_embedding_dim, tb_embedding_dim = tb_embedding_dim)

    def forward(self, ts_x, tb_x):

      bimodal_x = self.encoder(ts_x, tb_x)
      bimodal_x = self.ffn(bimodal_x,None, None)
      x = self.decoder(bimodal_x)

      return x
  
class HMP(nn.Module):
    def __init__(self, seq_len, n_features, d_model = 256, dropout = 0.9):
      super().__init__()


      self.seq_len = seq_len
      self.n_features = n_features
      self.ts_embedding_dim = d_model
      self.tb_embedding_dim = d_model
      self.sig = nn.Sigmoid()
      self.fc = nn.Linear(d_model, 1)
      self.ICU_Encoder = ICU_Encoder(seq_len = 48, n_features = self.n_features, d_model = d_model, dropout = dropout)
      self.ffn = FFN(d_model)
      self.dropout = nn.Dropout(dropout)
      self.pooler = MaxPoolLayer()
      self.relu1 = nn.ReLU()
      self.relu2 = nn.ReLU()
      self.relu3 = nn.ReLU()
      self.attn = SelfAttention(d_model)    
      self.t5 =  AutoModelForSeq2SeqLM.from_pretrained("LLM/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base").encoder
      for param in self.t5.parameters():
          param.requires_grad = False
      self.fc2 = nn.Linear(768, d_model)
      self.pooler = MaxPoolLayer()
      # self.relu_3 = nn.Relu()
    def forward(self, ts_x, tb_x):


      bimodal_x = self.ICU_Encoder(ts_x, tb_x)
      x = self.dropout(bimodal_x)
      x = self.fc(x)
      x = self.sig(x)
      return x



class ICU_Encoder(nn.Module):
    def __init__(self, seq_len, n_features, d_model, dropout):
      super().__init__()
      device = torch.device("cuda:1")
      self.ts_encoder = LSTM_Encoder(seq_len, n_features, d_model)
      self.tb_encoder = table_encoder(input_shape = 73, emb_shape = d_model)
      self.attn = SelfAttention(d_model)
      self.dropout = nn.Dropout(dropout)
      self.ffn = FFN(d_model)
      self.pooler = MaxPoolLayer()
      self.self_attn = SelfAttention(d_model)
    def forward(self,  ts_x, tb_x):
      

      hidden_emb, ts_x = self.ts_encoder(ts_x) # input: time series

      tb_x = self.tb_encoder(tb_x) # input:tabular data
      tb_x = torch.unsqueeze(tb_x, 1)
      ts_x = torch.unsqueeze(ts_x, 1)
      bimodal_x = torch.cat((ts_x,tb_x), -2)
      bimodal_x = self.self_attn(bimodal_x, None, None)
      bimodal_x = self.dropout(bimodal_x)
      bimodal_x = self.ffn(bimodal_x, None, None)
      bimodal_x = self.pooler(bimodal_x)
      
     
      return bimodal_x
  
class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size1, vocab_size2,  d_model = 256, dropout=0.1, dropout_emb=0.1, length=48):
        super().__init__()
        PATH = "LLM/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base"
        self.tokenizer = AutoTokenizer.from_pretrained(PATH)
        self.t5model = AutoModelForSeq2SeqLM.from_pretrained(PATH)
        self.embbedding1 = nn.Sequential(nn.Linear(vocab_size1, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding2 = nn.Sequential(nn.Linear(vocab_size2, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding3 = nn.Sequential(nn.Linear(3*d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding4 = nn.Sequential(nn.Linear(768, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.ICU_Encoder = ICU_Encoder(seq_len = 48, n_features = 1318, ts_embedding_dim = d_model, tb_embedding_dim = d_model)
        self.ffn_1 = FFN(d_model)
        self.ffn_2 = FFN(d_model)
        self.ffn_3 = FFN(d_model)
        self.ffn_4 = FFN(d_model)
        self.ffn_5 = FFN(d_model)
        self.attn = SelfAttention(d_model)
        self.d_model = d_model
        self.pooler = MaxPoolLayer()
        self.mask_icd = nn.Linear(1, d_model)
        self.mask_drug = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.output_mlp_1 = nn.Sequential(nn.Linear(d_model, vocab_size1), nn.Sigmoid())
        self.output_mlp_2 = nn.Sequential(nn.Linear(d_model, vocab_size2), nn.Sigmoid())
        self.device = torch.device("cuda:0")
        for param in self.t5model.parameters():
            param.requires_grad = False
    def forward(self, x1, x2, nums_icd, nums_drug, X,S, doc_emb, masked_icd, masked_drug):
        

        x1 = self.embbedding1(x1) # input: icd codes
        masked_x1 = self.embbedding1(masked_icd)+self.mask_icd(nums_icd) # input: icd codes

        x2 = self.embbedding2(x2) # input: drug codes
        masked_x2 = self.embbedding2(masked_drug)+self.mask_drug(nums_drug)
        ts_1, ts_2, ts_3 = X[:,0], X[:,1], X[:,2] # input: time series corresponding to each HADM(visit)
        tb_1, tb_2, tb_3 = S[:,0], S[:,1], S[:,2] # input: tabular data corresponding to each HADM(visit)


        doc_emb = self.embbedding4(doc_emb)

        icu_rep = torch.cat((self.ICU_Encoder(ts_1, tb_1), self.ICU_Encoder(ts_2, tb_2), self.ICU_Encoder(ts_3, tb_3)), -1)
        icu_rep = self.embbedding3(icu_rep)
        


        

        
        input_seqs_doc = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), icu_rep.unsqueeze(1)), 1)
        input_seqs_x1 = torch.cat((doc_emb.unsqueeze(1), x2.unsqueeze(1), icu_rep.unsqueeze(1)), 1)
        input_seqs_x2 = torch.cat((doc_emb.unsqueeze(1), x1.unsqueeze(1), icu_rep.unsqueeze(1)), 1)
        input_seqs_mcm_x1 = torch.cat((doc_emb.unsqueeze(1), masked_x1.unsqueeze(1), x2.unsqueeze(1), icu_rep.unsqueeze(1)), 1)
        input_seqs_mcm_x2 = torch.cat((doc_emb.unsqueeze(1), x1.unsqueeze(1), masked_x2.unsqueeze(1), icu_rep.unsqueeze(1)), 1)
       
    
        x = self.attn(input_seqs_doc, None, None)
        x = self.pooler(x)
        doc_rep = self.ffn_1(x, None, None)
        x = self.attn(input_seqs_x1, None, None)
        x = self.pooler(x)
        x1_rep = self.ffn_2(x, None, None)
        x = self.attn(input_seqs_x2, None, None)
        x = self.pooler(x)
        x2_rep = self.ffn_3(x, None, None)
        x = self.attn(input_seqs_mcm_x1, None, None)
        x = self.pooler(x)
        mcm_x1_rep = self.ffn_4(x, None, None)
        mcm_x1_rep = self.output_mlp_1(mcm_x1_rep)
        x = self.attn(input_seqs_mcm_x2, None, None)
        x = self.pooler(x)
        mcm_x2_rep = self.ffn_5(x, None, None)
        mcm_x2_rep = self.output_mlp_2(mcm_x2_rep)
        return doc_emb, doc_rep, x1, x1_rep, x2, x2_rep, mcm_x1_rep , mcm_x2_rep

class HADM_CLS(nn.Module):
    def __init__(self, vocab_size1, vocab_size2,  d_model = 256, dropout=0.7, dropout_emb=0.5, length=48):
        super().__init__()
        PATH = "LLM/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base"
        self.tokenizer = AutoTokenizer.from_pretrained(PATH)
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained("LLM/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base").encoder
        self.embbedding1 = nn.Sequential(nn.Linear(vocab_size1, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding2 = nn.Sequential(nn.Linear(vocab_size2, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding3 = nn.Sequential(nn.Linear(3*d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding4 = nn.Sequential(nn.Linear(768, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.ICU_Encoder = ICU_Encoder(seq_len = 48, n_features = 1318, d_model = d_model, dropout = dropout)
        self.ffn = FFN(d_model)
        self.attn = SelfAttention(d_model)
        self.d_model = d_model
        self.pooler = MaxPoolLayer()
        self.mask_icd = nn.Linear(1, d_model)
        self.mask_drug = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_mlp_admission = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.device = torch.device("cuda:0")
        for param in self.t5.parameters():
            param.requires_grad = False
    def forward(self, x1, x2, X,S, input_ids, attention_mask, token_type_ids):
        
        x1 = self.embbedding1(x1) # input: icd codes
        x2 = self.embbedding2(x2)  # input: drug codes
        ts_1, ts_2, ts_3 = X[:,0], X[:,1], X[:,2] # input: time series corresponding to each HADM(visit)
        tb_1, tb_2, tb_3 = S[:,0], S[:,1], S[:,2] # input: tabular data corresponding to each HADM(visit)

        
        text = self.t5(input_ids=input_ids,attention_mask=attention_mask,  return_dict=True).last_hidden_state


        doc_emb = torch.mean(text, dim=1)
        doc_emb = self.embbedding4(doc_emb)

        icu_rep = torch.cat((self.ICU_Encoder(ts_1, tb_1), self.ICU_Encoder(ts_2, tb_2), self.ICU_Encoder(ts_3, tb_3)), -1)
        icu_rep = self.embbedding3(icu_rep)
        

        input_seqs = torch.cat((doc_emb.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1), icu_rep.unsqueeze(1)), 1)
        
        x = self.attn(input_seqs, None, None)
        x = self.pooler(x)
        rep = self.ffn(x, None, None)
        rep = self.dropout(rep)
        return self.output_mlp_admission(rep)


class T5_CLS(nn.Module):
    def __init__(self, vocab_size1, vocab_size2,  d_model = 256, dropout=0.7, dropout_emb=0.5, length=48):
        super().__init__()
        PATH = "LLM/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base"
        self.tokenizer = AutoTokenizer.from_pretrained(PATH)
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained("LLM/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base").encoder
        self.embbedding1 = nn.Sequential(nn.Linear(vocab_size1, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding2 = nn.Sequential(nn.Linear(vocab_size2, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding3 = nn.Sequential(nn.Linear(3*d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.embbedding4 = nn.Sequential(nn.Linear(768, 1), nn.LayerNorm(d_model), nn.ReLU())
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.ICU_Encoder = ICU_Encoder(seq_len = 48, n_features = 1318, d_model = d_model, dropout = dropout)
        self.ffn = FFN(d_model)
        self.attn = SelfAttention(d_model)
        self.d_model = d_model
        self.pooler = MaxPoolLayer()
        self.mask_icd = nn.Linear(1, d_model)
        self.mask_drug = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_mlp_admission = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.device = torch.device("cuda:0")
    def forward(self, input_ids, attention_mask, token_type_ids):
        

        text = self.t5(input_ids=input_ids,attention_mask=attention_mask,  return_dict=True).last_hidden_state


        doc_emb = torch.mean(text, dim=1)

        return self.output_mlp_admission(rep)

    
    
class HADM_AE(nn.Module):
    def __init__(self, vocab_size1, vocab_size2,  d_model, dropout=0.1, dropout_emb=0.1, length=48):
        super().__init__()
        self.enc = Transformer_Encoder(vocab_size1, vocab_size2,  d_model, dropout=0.1, dropout_emb=0.1, length=48)
    def forward(self, x1, x2, nums_icd, nums_drug, X,S, text, masked_icd, masked_drug):
        return self.enc(x1, x2, nums_icd, nums_drug, X,S, text, masked_icd, masked_drug)






