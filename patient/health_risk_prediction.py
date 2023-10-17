import argparse
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from torch.optim import *
from sklearn.metrics import precision_recall_curve, auc
from models.og_dataset import *
from models.baseline import *
from utils.utils import check_path, export_config, bool_flag
import csv
import os
from datetime import datetime
from models.adacare import AdaCare

def eval_metric(eval_set, model, encoder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(eval_set):
            labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
            if encoder == 'adacare':
              logits = model(ehr, device)
            else:
            
              logits = model(ehr, mask, lengths, time_step, code_mask)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            score = scores[:, 1]
            pred = scores.argmax(1)
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

    return accuary, precision, recall, f1, roc_auc, pr_auc, kappa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int)
    parser.add_argument('--encoder', default='hita', choices=['hita', 'lstm',  'gruself',  'retain', 'adacare'])
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--dropout', default=0.8, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.8, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=2, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--max_len', default=50, type=int, help='max visits of EHR')
    parser.add_argument('--max_num_codes', default=20, type=int, help='max number of ICD codes in each visit')
    parser.add_argument('--max_num_blks', default=120, type=int, help='max number of blocks in each visit')
    parser.add_argument('--blk_emb_path', default='./data/processed/block_embedding.npy',
                        help='embedding path of blocks')
    parser.add_argument('--blk_vocab_path', default='./data/processed/block_vocab.txt')
    parser.add_argument('--target_disease', default='Heart_failure', choices=['Heart_failure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic'])
    parser.add_argument('--target_att_heads', default=4, type=int, help='target disease attention heads number')
    parser.add_argument('--mem_size', default=20, type=int, help='memory size')
    parser.add_argument('--mem_update_size', default=15, type=int, help='memory update size')
    parser.add_argument('-lr', '--learning_rate', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred','gen'], help='run training or evaluation')
    parser.add_argument('--save_dir', default='./saved_models/', help='model output directory')
    parser.add_argument('--pretrain', default=False, help='flag for using pretrained model')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)
    path = "patient_level/"
    files = os.listdir(path)
    print(args)
    if str(args.pretrain) + '_' + str(args.target_disease) + '_' + str(args.encoder)  +  '_' + '.csv' in files:
        print("conducted_experiments")
    else:
        config_path = os.path.join(args.save_dir, 'config.json')
        model_path = os.path.join(args.save_dir, 'model.pt')
        log_path = os.path.join(args.save_dir, 'log.csv')
        export_config(args, config_path)
        check_path(model_path)
        with open(log_path, 'w') as fout:
            fout.write('step,train_auc,dev_auc,test_auc\n')
    
        blk_emb = np.load(args.blk_emb_path)
        blk_pad_id = len(blk_emb) - 1
        if args.target_disease == 'Heart_failure':
            code2id = pickle.load(open('./data/hf/hf_code2idx.pickle', 'rb'))
            pad_id = len(code2id)+1
            data_path = './data/hf/hf'
            emb_path = './data/processed/heart_failure.npy'
        elif args.target_disease == 'COPD':
            code2id = pickle.load(open('./data/copd/copd_code2idx.pickle', 'rb'))
            pad_id = len(code2id)+1
            data_path = './data/copd/copd'
            emb_path = './data/processed/COPD.npy'
        elif args.target_disease == 'Kidney':
            code2id = pickle.load(open('./data/kidney/kidney_code2idx.pickle', 'rb'))
            pad_id = len(code2id)+1
            data_path = './data/kidney/kidney'
            emb_path = './data/processed/kidney_disease.npy'
        elif args.target_disease == 'Dementia':
            code2id = pickle.load(open('./data/dementia/dementia_code2idx.pickle', 'rb'))
            pad_id = len(code2id)+1
            data_path = './data/dementia/dementia'
            emb_path = './data/processed/dementia.npy'
        elif args.target_disease == 'Amnesia':
            code2id = pickle.load(open('./data/amnesia/amnesia_code2idx.pickle', 'rb'))
            pad_id = len(code2id)+1
            data_path = './data/amnesia/amnesia'
            emb_path = './data/processed/amnesia.npy'
        elif args.target_disease == 'mimic':
            code2id = pickle.load(open('./data/mimic/mimic_code2idx_sps.pickle', 'rb'))
            pad_id = len(code2id)+1
            data_path = './data/mimic/mimic'
            emb_path = './data/processed/mimic.npy'
        else:
            raise ValueError('Invalid disease')
        
        torch.manual_seed(args.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        if args.target_disease == 'mimic':
            train_dataset = MyDataset3(data_path + '_training_sps.pickle',
                                      args.max_len, args.max_num_codes, args.max_num_blks, pad_id, device)
            dev_dataset = MyDataset3(data_path + '_validation_sps.pickle',
                                    args.max_len,
                                    args.max_num_codes, args.max_num_blks, pad_id, device)
            test_dataset = MyDataset3(data_path + '_testing_sps.pickle', args.max_len,
                                     args.max_num_codes, args.max_num_blks, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
        else:
            train_dataset = MyDataset2(data_path + '_training_sps.pickle', data_path + '_training_txt.pickle',
                                      args.max_len, args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
            dev_dataset = MyDataset2(data_path + '_validation_sps.pickle', data_path + '_validation_txt.pickle', args.max_len,
                                    args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
            test_dataset = MyDataset2(data_path + '_testing_sps.pickle', data_path + '_testing_txt.pickle', args.max_len,
                                     args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    
        if args.encoder == 'hita':
            model = HitaNet(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                     args.max_len, pretrain = args.pretrain)
        elif args.encoder == 'lstm':
            model = LSTM_encoder(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                     args.max_len, pretrain = args.pretrain)
        elif args.encoder == 'gruself':
            model = GRUSelf(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len, pretrain = args.pretrain)
        elif args.encoder == 'retain':
            model = Retain(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len, pretrain = args.pretrain)
        elif args.encoder == 'adacare':
          model = AdaCare(pad_id, hidden_dim=args.d_model, kernel_size=2, kernel_num=64, input_dim=args.d_model, output_dim=1, dropout=args.dropout, r_v=4,
                     r_c=4, activation='sigmoid', pretrain = args.pretrain)
        else:
            raise ValueError('Invalid encoder')
        model.to(device)
    
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.learning_rate}
        ]
        optim = Adam(grouped_parameters)
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    
        print('parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
            else:
                print('\t{:45}\tfixed\t{}'.format(name, param.size()))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\ttotal:', num_params)
        print()
        print('-' * 71)
        global_step, best_dev_epoch = 0, 0
        best_dev_auc, final_test_auc, total_loss = 0.0, 0.0, 0.0
        best_kappa, best_f1 = 0,0
        model.train()
        for epoch_id in range(args.n_epochs):
            print('epoch: {:5} '.format(epoch_id))
            model.train()
            start_time = time.time()
            for i, data in enumerate(train_dataloader):
                labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
                optim.zero_grad()
                if args.encoder == 'adacare':
                  outputs = model(ehr, device)
                else:
                  outputs = model(ehr, mask, lengths, time_step, code_mask)
                loss = loss_func(outputs, labels)
                loss.backward()
                total_loss += (loss.item() / labels.size(0)) * args.batch_size
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                   total_loss,
                                                                                   ms_per_batch))
                    total_loss = 0.0
                    start_time = time.time()
                global_step += 1
    
            model.eval()
            train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa = eval_metric(train_dataloader, model, args.encoder)
            dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa = eval_metric(dev_dataloader, model, args.encoder)
            test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model, args.encoder)
            print('-' * 71)
            print('| step {:5} | train_acc {:7.4f} | dev_acc {:7.4f} | test_acc {:7.4f} '.format(global_step,
                                                                                                 train_acc,
                                                                                                 dev_acc,
                                                                                                 test_acc))
            print(
                '| step {:5} | train_precision {:7.4f} | dev_precision {:7.4f} | test_precision {:7.4f} '.format(
                    global_step,
                    tr_precision,
                    d_precision,
                    t_precision))
            print('| step {:5} | train_recall {:7.4f} | dev_recall {:7.4f} | test_recall {:7.4f} '.format(
                global_step,
                tr_recall,
                d_recall,
                t_recall))
            print('| step {:5} | train_f1 {:7.4f} | dev_f1 {:7.4f} | test_f1 {:7.4f} '.format(global_step,
                                                                                              tr_f1,
                                                                                              d_f1,
                                                                                              t_f1))
            print('| step {:5} | train_auc {:7.4f} | dev_auc {:7.4f} | test_auc {:7.4f} '.format(global_step,
                                                                                                 tr_roc_auc,
                                                                                                 d_roc_auc,
                                                                                                 t_roc_auc))
            print('| step {:5} | train_pr {:7.4f} | dev_pr {:7.4f} | test_pr {:7.4f} '.format(global_step,
                                                                                                 tr_pr_auc,
                                                                                                 d_pr_auc,
                                                                                                 t_pr_auc))
            print('| step {:5} | train_kappa {:7.4f} | dev_kappa {:7.4f} | test_kappa {:7.4f} '.format(global_step,
                                                                                                 tr_kappa,
                                                                                                 d_kappa,
                                                                                                 t_kappa))
            print('-' * 71)
    
            if d_f1 >= best_dev_auc:
                best_dev_auc = d_f1
                final_test_auc = t_pr_auc
                best_dev_epoch = epoch_id
                best_f1 = t_f1
                best_kappa = t_kappa
                torch.save([model, args], model_path)
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{},{}, {},{}\n'.format(global_step, tr_pr_auc, d_pr_auc, t_pr_auc,  t_f1, t_kappa))
                print(f'model saved to {model_path}')
            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
    
                break
        
        now = datetime.now()
        results_file = open(path+str(args.pretrain) + '_' + str(args.target_disease) + '_' + str(args.encoder)  +  '_' + '.csv','w',encoding  = 'gbk')
        csv_w = csv.writer(results_file)
        csv_w.writerow([final_test_auc,  best_f1, best_kappa]) 
        print()
        print('training ends in {} steps'.format(global_step))
        print('best dev auc: {:.4f} (at epoch {})'.format(best_dev_auc, best_dev_epoch))
        print('final test auc: {:.4f}'.format(final_test_auc))
        print(final_test_auc,  best_f1, best_kappa)
        print()
        results_file.close()


def pred(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()
    blk_emb = np.load(old_args.blk_emb_path)
    blk_pad_id = len(blk_emb) - 1
    if old_args.target_disease == 'Heart_failure':
        code2id = pickle.load(open('./data/hf/hf_code2idx.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/hf/hf_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/hf/hf'
    elif old_args.target_disease == 'COPD':
        code2id = pickle.load(open('./data/copd/copd_code2idx.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/copd/copd_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/copd/copd'
    elif old_args.target_disease == 'Kidney':
        code2id = pickle.load(open('./data/kidney/kidney_code2idx.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/kidney/kidney_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/kidney/kidney'
    elif old_args.target_disease == 'Amnesia':
        code2id = pickle.load(open('./data/amnesia/amnesia_code2idx.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/amnesia/amnesia_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/amnesia/amnesia'
    elif old_args.target_disease == 'Dementia':
        code2id = pickle.load(open('./data/dementia/dementia_code2idx.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/dementia/dementia_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/dementia/dementia'
    else:
        raise ValueError('Invalid disease')
    dev_dataset = MyDataset(data_path + '_validation_sps.pickle', data_path + '_validation_txt.pickle',
                            old_args.max_len, old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
    test_dataset = MyDataset(data_path + '_testing_sps.pickle', data_path + '_testing_txt.pickle', old_args.max_len,
                             old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    # dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc = eval_metric(dev_dataloader, model)
    test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model)
    log_path = os.path.join(args.save_dir, 'result.csv')
    with open(log_path, 'w') as fout:
        fout.write('test_auc,test_f1,test_pre,test_recall,test_pr_auc,test_kappa\n')
        fout.write(
            '{},{},{},{},{},{}\n'.format(t_roc_auc, t_f1, t_precision, t_recall, t_pr_auc, t_kappa))
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(test_dataloader):
            labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
            logits = model(ehr, mask, lengths, time_step, code_mask)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            score = scores[:, 1]
            pred = scores.argmax(1)
            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, pred))
            y_score = np.concatenate((y_score, score))
        with open(os.path.join(args.save_dir, 'prediction.csv'), 'w') as fout2:
            fout2.write('prediciton,score,label\n')
            for i in range(len(y_true)):
                fout2.write('{},{},{}\n'.format(y_pred[i], y_score[i], y_true[i]))


if __name__ == '__main__':
    main()
    globals().clear()