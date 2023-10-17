import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from operations import *
import model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class LSTM_bimodal(nn.Module):
    def __init__(self, vocab_size1, vocab_size2,  d_model = 256, dropout=0.5, dropout_emb=0.5, length=48, pretrain = False):
        super().__init__()
        self.embbedding1 = nn.Sequential(nn.Linear(vocab_size1, d_model), nn.ReLU())
        self.linear = nn.Linear(vocab_size2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 1))
        self.pooler = MaxPoolLayer()
        if pretrain:
            self.rnns = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        else:
            self.rnns = nn.LSTM(vocab_size1, d_model, 1, bidirectional=False, batch_first=True)
        self.sig = nn.Sigmoid()
        self.ts_encoder = model.LSTM_Encoder(length, vocab_size1, d_model)
        self.linear_2 = nn.Linear(32, d_model)
        self.pretrain = pretrain


    def forward(self, x):
        if self.pretrain == True:
            x = self.ts_encoder(x)[0]
            x = self.emb_dropout(x)

        rnn_output, _ = self.rnns(x)
        x = self.pooler(rnn_output)
        x = self.output_mlp(x)
        x = self.sig(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, d_model, dropout=0.5, dropout_emb=0.5, length=48, pretrain = False):
        super().__init__()
        self.embbedding1 = nn.Sequential(nn.Linear(vocab_size1, d_model), nn.ReLU())
        self.linear = nn.Linear(vocab_size2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 1))
        self.pooler = MaxPoolLayer()
        self.attention = SelfAttention(d_model)
        self.ffn = FFN(d_model)
        self.sig = nn.Sigmoid()
        self.ts_encoder = model.LSTM_Encoder(length, vocab_size1, d_model)
        self.linear_2 = nn.Linear(32, d_model)
        self.pretrain = pretrain
    def forward(self, x):
        if self.pretrain == True:
            x = self.ts_encoder(x)[0]
            x = self.emb_dropout(x)
        else:
            x = self.embbedding1(x)

        x = self.attention(x, None, None)
        x = self.ffn(x, None, None)
        x = self.dropout(x)
        x = self.pooler(x)
        x = self.output_mlp(x)
        x = self.sig(x)
        return x
    
class ClinicalT5(nn.Module):
    def __init__(self, d_model = 256):
      super().__init__()

      self.sig = nn.Sigmoid()
      self.t5 =  AutoModelForSeq2SeqLM.from_pretrained("LLM/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base").encoder
      self.fc2 = nn.Linear(768, 1)
      self.pooler = MaxPoolLayer()
      self.relu1 = nn.ReLU()
    def forward(self, ts_x, tb_x, input_ids, attention_mask):
      
      text = self.t5(input_ids=input_ids,attention_mask=attention_mask, return_dict=True).last_hidden_state
      sent_emb = torch.mean(text, dim=1)
      sent_emb = self.fc2(sent_emb)
      x = self.sig(sent_emb)
      return x

class Raim(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, vocab_size3, d_model, dropout=0.1, dropout_emb=0.1, length=48):
        super().__init__()
        self.embbedding1 = nn.Sequential(nn.Linear(vocab_size1, d_model), nn.ReLU())
        self.embbedding2 = nn.Sequential(nn.Linear(vocab_size2, d_model), nn.ReLU())
        self.linear = nn.Linear(vocab_size3, d_model)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()

        self.hidden_size = d_model

        self.rnn = nn.LSTM(d_model, d_model, 2, dropout=0.5)
        self.attn = nn.Linear(10, 10)
        self.attn1 = nn.Linear(60, 10)

        self.dense_h = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=1)
        self.hidden2label = nn.Linear(d_model, 1)
        self.grucell = nn.GRUCell(d_model, d_model)

        self.mlp_for_x = nn.Linear(d_model, 1, bias=False)
        self.mlp_for_hidden = nn.Linear(d_model, length, bias=True)
        
        self.sigmoid = nn.Sigmoid()


    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

    def forward(self, x1, x2, s):
        x1 = self.embbedding1(x1)
        x2 = self.embbedding2(x2)
        s = self.linear(s)
        input_seqs = x1 + x2
        x = input_seqs
        self.hidden = self.init_hidden(x.size(0)).to(x.device)
        for i in range(x.size(1)):
            tt = x[:, 0:i + 1, :].reshape(x.size(0), (i + 1) * x[:, 0:i + 1, :].shape[2])
            if i < x.size(1) - 1:
                padding = torch.zeros(x.size(0), x.size(1)*x.size(2) - tt.shape[1]).to(x.device)
                self.temp1 = torch.cat((tt, padding), 1)
            else:
                self.temp1 = tt

            self.input_padded = self.temp1.reshape(x.size(0), x.size(1), x.size(-1))

            #### multuply with guidance #######
            temp_guidance = torch.zeros(x.size(0), x.size(1), 1).to(x.device)

            # temp_guidance[:, 0:i + 1, :] = x2[:, 0:i + 1, 0].unsqueeze(-1)

            if i > 0:

                zero_idx = torch.where(torch.sum(x2[:, :i, 0], dim=1) == 0)
                if len(zero_idx[0]) > 0:
                    temp_guidance[zero_idx[0], :i, 0] = 1

            temp_guidance[:, i, :] = 1

            self.guided_input = torch.mul(self.input_padded, temp_guidance)

            ######### MLP ###########
            self.t1 = self.mlp_for_x(self.guided_input) + self.mlp_for_hidden(self.hidden).reshape(x.size(0), x.size(1), 1)

            ######### softmax-> multiply->  context vector ###########
            self.t1_softmax = self.softmax(self.t1)
            final_output = torch.mul(self.input_padded, self.t1_softmax)

            context_vec = torch.sum(final_output, dim=1)

            self.hx = self.grucell(context_vec, self.hidden)
            self.hidden = self.hx

        y = self.hidden2label(self.hidden + s)
        return self.sigmoid(y)



class DCMN(nn.Module):

    def __init__(self, vocab_size1, vocab_size2, vocab_size3, d_model, dropout=0.1, dropout_emb=0.1, length=48):
        super().__init__()
        self.embbedding1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=5),
                                         nn.ReLU(),
                                         nn.Linear((vocab_size1 - 10) // 5 + 1, d_model))
        self.embbedding2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=5),
                                         nn.ReLU(),
                                         nn.Linear((vocab_size2 - 10) // 5 + 1, d_model))
        self.linear = nn.Linear(vocab_size3, d_model)
        self.batchnorm1 = nn.BatchNorm1d(d_model)
        self.batchnorm2 = nn.BatchNorm1d(d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 1))
        self.c_emb = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.c_out = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.w_emb = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.w_out = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.linear4 = nn.Linear(d_model, d_model)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_linear2 = nn.Linear(d_model, d_model)
        self.pooler = MaxPoolLayer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, s):
        bs, l, fdim = x1.size()
        x1 = x1.view(bs * l, -1).unsqueeze(1)
        x2 = x2.view(bs * l, -1).unsqueeze(1)
        x1 = self.embbedding1(x1)
        x2 = self.embbedding2(x2)
        x1 = x1.squeeze().view(bs, l, -1)
        x2 = x2.squeeze().view(bs, l, -1)
        s = self.dropout(self.linear(s))
        x1 = self.batchnorm1(x1.permute(0, 2, 1)).permute(0, 2, 1)
        x2 = self.batchnorm2(x2.permute(0, 2, 1)).permute(0, 2, 1)
        wm_embedding_memory, _ = self.w_emb(x1)
        wm_out_query, _ = self.w_out(x1)
        cm_embedding_memory, _ = self.c_emb(x2)
        cm_out_query, _ = self.c_out(x2)
        wm_in = cm_out_query[:, -1]
        cm_in = wm_out_query[:, -1]
        w_embedding_E = self.linear1(wm_embedding_memory)
        w_embedding_F = self.linear2(wm_embedding_memory)
        wm_out = torch.matmul(wm_in.unsqueeze(1), w_embedding_E.permute(0, 2, 1))
        wm_prob = torch.softmax(wm_out, dim=-1)
        wm_contex = torch.matmul(wm_prob, w_embedding_F)
        wm_gate_prob = torch.sigmoid(self.gate_linear(wm_in)).unsqueeze(1)
        wm_dout = wm_contex * wm_gate_prob + wm_in.unsqueeze(1) * (1 - wm_gate_prob)

        c_embedding_E = self.linear3(cm_embedding_memory)
        c_embedding_F = self.linear4(cm_embedding_memory)
        cm_out = torch.matmul(cm_in.unsqueeze(1), c_embedding_E.permute(0, 2, 1))
        cm_prob = torch.softmax(cm_out, dim=-1)
        cm_contex = torch.matmul(cm_prob, c_embedding_F)
        cm_gate_prob = torch.sigmoid(self.gate_linear2(cm_in)).unsqueeze(1)
        cm_dout = cm_contex * cm_gate_prob + cm_in.unsqueeze(1) * (1 - cm_gate_prob)
        output = wm_dout + cm_dout
        output = self.output_mlp(output.squeeze() + s)
        return self.sigmoid(output)


class Mufasa(nn.Module):

    def __init__(self, vocab_size1, vocab_size2, vocab_size3, d_model, dropout=0.1, dropout_emb=0.1, length=48):
        super().__init__()
        self.embbedding1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=5),
                                         nn.ReLU(),
                                         nn.Linear((vocab_size1 - 10) // 5 + 1, d_model))
        self.embbedding2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=5),
                                         nn.ReLU(),
                                         nn.Linear((vocab_size2 - 10) // 5 + 1, d_model))
        self.linear = nn.Linear(vocab_size3, d_model)
        self.linear_conti = nn.Linear(d_model, d_model)
        self.linear_cate = nn.Linear(2*d_model, d_model)
        self.linears = nn.Linear(2 * d_model, d_model)
        self.linear_late = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=False))
        self.dense = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(inplace=False), nn.Linear(4*d_model, d_model))
        self.relu = nn.ReLU(inplace=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.self_att = SelfAttention(d_model)
        self.self_att2 = SelfAttention(d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.leaky = nn.LeakyReLU(inplace=False)
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2, s):
        bs, l, fdim = x1.size()
        x1 = x1.view(bs * l, -1).unsqueeze(1).clone()
        x2 = x2.view(bs * l, -1).unsqueeze(1).clone()
        x1 = self.embbedding1(x1)
        x2 = self.embbedding2(x2)
        x1 = x1.squeeze().view(bs, l, -1)
        x2 = x2.squeeze().view(bs, l, -1)
        s = self.linear(s)
        continues_res = x2
        continues_hs = self.layernorm(x2)
        continues_hs = self.self_att(continues_hs, None, None)
        continues_hs = self.leaky(continues_hs)
        continues_hs = continues_res + continues_hs
        continuous_res = continues_hs
        continues_hs = self.layernorm(continues_hs)
        continues_hs = self.linear_conti(continues_hs)
        continues_hs = self.relu(continues_hs)
        continues_hs = continuous_res + continues_hs
        categorical_res = x1
        categorical_hs = self.layernorm2(x1)
        categorical_hs = self.self_att2(categorical_hs, None, None)
        categorical_hs = torch.cat((categorical_hs, categorical_res), dim=-1)
        categorical_res = categorical_hs.clone()
        categorical_hs = self.linear_cate(categorical_hs)
        categorical_hs = self.relu(categorical_hs)
        categorical_res = self.linears(categorical_res)
        categorical_hybrid_point = categorical_hs + categorical_res
        categorical_late_point = self.linear_late(categorical_res)
        temp = s.unsqueeze(1).clone()
        fusion_hs = temp.expand_as(categorical_hybrid_point) + categorical_hybrid_point
        fusion_res = fusion_hs
        fusion_hs = self.layernorm3(fusion_hs)
        fusion_branch = self.conv(fusion_hs.permute(0, 2, 1)).permute(0, 2, 1)
        out = fusion_res + fusion_hs + fusion_branch + categorical_late_point + continues_hs
        out = self.pooler(out)
        out = self.output_mlp(out)
        return self.sigmoid(out)

if __name__ == '__main__':
    model = Transformer(1318, 73, 256)
    x1 = torch.randn((32, 48, 1318))
    s = torch.randn((32, 73))
    print(model(x1, s).size())