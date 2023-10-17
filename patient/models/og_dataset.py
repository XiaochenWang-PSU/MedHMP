import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def padMatrix(input_data, max_num_pervisit, maxlen, pad_id):
    pad_id = torch.zeros(7687)
    #pad_seq = torch.tensor([pad_id] * max_num_pervisit)
    output = []
    lengths = []
    for seq in input_data:
        record_ids = seq
        record_ids = record_ids[-maxlen:]
        lengths.append(len(record_ids))
        for j in range(0, (maxlen - len(record_ids))):
            record_ids.append(pad_id)
        output.append(record_ids)
    masks = []
    for l in lengths:
        mask = np.tril(np.ones((maxlen, maxlen)))
        # mask[:l, :l] = np.tril(np.ones((l, l)))
        masks.append(mask)
    return output, masks, lengths


def padMatrix2(input_data, max_num_pervisit, maxlen, pad_id):
    pad_seq = [pad_id] * max_num_pervisit
    output = []
    masks = []
    for seq in input_data:
        record_ids = []
        mask = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            mask_v = [1] * len(visit_ids)
            for i in range(0, (max_num_pervisit - len(visit_ids))):
                visit_ids.append(pad_id)
                mask_v.append(0)
            record_ids.append(visit_ids)
            mask.append(mask_v)
        record_ids = record_ids[-maxlen:]
        mask = mask[-maxlen:]
        for j in range(0, (maxlen - len(record_ids))):
            record_ids.append(pad_seq)
            mask.append([0] * max_num_pervisit)
        output.append(record_ids)
        masks.append(mask)
    return output, masks


def padTime(time_step, maxlen, pad_id):
    for k in range(len(time_step)):
        time_step[k] = time_step[k][-maxlen:]
        while len(time_step[k]) < maxlen:
            time_step[k].append(pad_id)
    return time_step

def codeMask(input_data, max_num_pervisit, maxlen):
    batch_mask = np.zeros((len(input_data), maxlen, max_num_pervisit), dtype=np.float32) + 1e+20
    output = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        output.append(record_ids)

    for bid, seq in enumerate(output):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_mask[bid, pid, tid] = 0
    return batch_mask
class MyDataset2(Dataset):
    def __init__(self, dir_ehr, dir_txt, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id, txt_pad_id,
                 device):
        ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _ = None
        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.LongTensor(self.ehr[idx]).to(
            self.device), \
               torch.LongTensor(self.mask_ehr[idx]).to(self.device), _, \
               _, torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device), \
               torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)
class MyDataset3(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id,
                 device):
        ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        #txt = pickle.load(open(dir_txt, 'rb'))
        self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        #self.txt, self.mask_txt = padMatrix2(txt, max_numblk_pervisit, max_len, txt_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
  
        #return idx
        #assert torch.LongTensor(self.mask_txt[idx]).size() == torch.LongTensor(self.txt[idx]).size()
 


        _ = None
        ehr_idx = torch.stack(self.ehr[idx])
        #ehr_idx = np.array(self.ehr[idx]).astype(float)
        # ehr_idx = torch.from_numpy(ehr_idx)
        #print(torch.LongTensor(self.ehr[idx]).to(self.device))
        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), ehr_idx.to(self.device), torch.Tensor(self.mask_ehr[idx]).to(self.device), _, \
               _, torch.tensor(self.lengths[idx],dtype=torch.long).to(self.device), \
               torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)

class MyDataset(Dataset):
    def __init__(self, dir_ehr, dir_txt, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id, txt_pad_id,
                 device):
        ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        txt = pickle.load(open(dir_txt, 'rb'))
        self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.txt, self.mask_txt = padMatrix2(txt, max_numblk_pervisit, max_len, txt_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
  
        #return idx
        #assert torch.LongTensor(self.mask_txt[idx]).size() == torch.LongTensor(self.txt[idx]).size()
 


          
        ehr_idx = torch.stack(self.ehr[idx])
        #ehr_idx = np.array(self.ehr[idx]).astype(float)
        # ehr_idx = torch.from_numpy(ehr_idx)
        #print(torch.LongTensor(self.ehr[idx]).to(self.device))
        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), ehr_idx.to(self.device), torch.Tensor(self.mask_ehr[idx]).to(self.device), torch.LongTensor(self.txt[idx]).to(self.device), \
               torch.LongTensor(self.mask_txt[idx]).to(self.device), torch.tensor(self.lengths[idx],
                                                                                  dtype=torch.long).to(self.device), \
               torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)

def collate_fn(batch):
    label, ehr, mask, txt, mask_txt, length, time_step, code_mask = [], [], [], [], [], [], [], []
    _=None
    for data in batch:
        label.append(data[0])
        ehr.append(data[1])
        mask.append(data[2])
        txt.append(data[3])
        mask_txt.append(data[4])
        length.append(data[5])
        time_step.append(data[6])
        code_mask.append(data[7])
    return torch.stack(label, 0), torch.stack(ehr, 0), torch.stack(mask, 0), _, \
           _, torch.stack(length, 0), torch.stack(time_step, 0), torch.stack(code_mask, 0)
