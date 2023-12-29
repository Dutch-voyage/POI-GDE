import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import csv

import os, torch, random, logging
import math
import time
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from models.EmbeddingLayer import EmbeddingLayer
from models.MultiFuser import MultiFuser
from models.GeoEmbedding import GeoEmbedding
from models.TimeEmbedding import TimeEmbedding

from models.ODE import NeuralODE, GraphODEF
from models.Differentiate import Differentiate

from dataloaders.dataloader_ode import MultiSessionsGraph
from Metrics.metrics import hit_rate, reciprocal_rank, SortLoss
from utils.Scheduler import GradualWarmupScheduler

class ODETrainer:
    def __init__(self, **kwargs):
        # torch.multiprocessing.set_sharing_strategy('file_system')
        self.data = kwargs['data']
        self.log = kwargs['log']

        self.kwargs = kwargs
        self.if_resume = kwargs['if_resume']
        self.ckpt_path = kwargs['ckpt_path']
        self.if_pretrain_fuse = kwargs['if_pretrain_fuse']
        self.if_pretrain_truth = kwargs['if_pretrain_truth']
        self.pretrain_fuse_path = kwargs['pretrain_fuse_path']
        self.pretrain_truth_path = kwargs['pretrain_truth_path']
        self.device = torch.device(f'cuda:{kwargs["gpu"]}')
        # self.device = torch.device('cpu')
        self.start_epoch = 1
        self.epoch = kwargs['epoch']
        self.lr = kwargs['lr']
        self.multiplier = kwargs['multiplier']
        self.weight_decay = kwargs['weight_decay']
        self.T_max = kwargs['T_max']
        self.eta_min = kwargs['eta_min']
        self.batch = kwargs['batch']
        self.patience = kwargs['patience']
        self.graph_size = kwargs['window_size']
        self.embed = kwargs['embed']
        self.dropout = kwargs['dropout']
        self.seed_torch(kwargs['seed'])
        self.epoch_info = []  # 存epoch_info  偶数步 recall2 20 mrr 2 20, 每次复原打印（从左到右）
        self.info_step = kwargs['info_step']
        self.loss_weight = kwargs['loss_weight']
        self.best_res = []
        LOG_FORMAT = "%(asctime)s  %(message)s"
        DATE_FORMAT = "%m/%d %H:%M"

        console_handler = logging.StreamHandler()  # 输出到控制台
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        self.LOGGER = logging.getLogger(__name__)
        self.LOGGER.addHandler(console_handler)

        if self.log is not None:
            log_name = self.log + time.strftime(
                '%Y-%m-%d-%H-%M-%S') + '_TKY_new' + f'_{self.data}_ ' + f'{self.if_resume}' + '.log'
            file_handler = logging.FileHandler(log_name)  # 输出到文件
            self.LOGGER.addHandler(file_handler)

        self.Latsup, self.Latinf, self.Lonsup, self.Loninf, \
        self.Ysup, self.Yinf, self.Msup, self.Minf, self.Wsup, self.Winf, self.Dsup, self.Dinf, self.Hsup, self.Hinf, \
        self.locations, self.n_user, self.n_poi, self.n_cid, self.cid_list = self.load_data()

        self.train_set = MultiSessionsGraph(device=self.device, window_size=self.graph_size, root=f'data/{self.data}',
                                            phrase='train')
        # self.val_set = MultiSessionsGraph(device=self.device, root=f'processed_data/{self.data}',
        #                                   phrase='val')
        self.test_set = MultiSessionsGraph(device=self.device, window_size=self.graph_size, root=f'data/{self.data}',
                                           phrase='test')

        self.LOGGER.info(f'Data loaded.')
        self.LOGGER.info(f'user: {self.n_user}\t poi: {self.n_poi}')
        self.LOGGER.info(self.kwargs)

        self.Poi_embeds = EmbeddingLayer(self.n_poi + self.n_user, self.embed).to(self.device)
        self.Geo_embeds = GeoEmbedding(self.Latsup, self.Latinf, self.Lonsup, self.Loninf, self.embed, self.device).to(
            self.device)
        self.User_embeds = EmbeddingLayer(self.n_user, self.embed).to(self.device)
        self.Category_embeds = EmbeddingLayer(self.n_cid + self.n_user, self.embed).to(self.device)
        self.Time_embeds = TimeEmbedding(self.Msup, self.Minf, self.Wsup, self.Winf,
                                         self.Dsup, self.Dinf, self.Hsup, self.Hinf,
                                         self.embed, self.device).to(self.device)

        self.Poi_proj = MultiFuser(self.embed, 1).to(self.device)
        self.Cat_proj = MultiFuser(self.embed, 1).to(self.device)
        self.Geo_proj = MultiFuser(self.embed, 1).to(self.device)

        self.GraphODEF = GraphODEF(self.Time_embeds, self.graph_size, self.embed, self.device).to(self.device)
        self.NeuralODE = NeuralODE(self.GraphODEF, self.n_poi, self.kwargs['max_step']).to(self.device)

        # self.Differentiate = Differentiate(self.embed).to(self.device)

        if self.if_pretrain_fuse:
            self.load_pretrain_fuse()
        if self.if_pretrain_truth:
            self.load_pretrain_truth()

        self.scaler = torch.cuda.amp.GradScaler()

        self.optimizer = torch.optim.Adam([
            {'params': self.Geo_proj.parameters()},
            {'params': self.Cat_proj.parameters()},
            {'params': self.Poi_proj.parameters()},
            {'params': self.Geo_embeds.parameters()},
            {'params': self.Category_embeds.parameters()},
            {'params': self.User_embeds.parameters()},
            {'params': self.Poi_embeds.parameters()},
            {'params': self.NeuralODE.parameters()},
        ],
            lr=self.lr, weight_decay=self.weight_decay)
        cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  #
            optimizer=self.optimizer, T_max=self.T_max, eta_min=self.eta_min)
        warmUpScheduler = GradualWarmupScheduler(optimizer=self.optimizer, multiplier=self.multiplier,
                                                 warm_epoch=15, after_scheduler=cosineScheduler)
        self.scheduler = warmUpScheduler
        print('use scheduler')  if self.kwargs['is_scheduler'] else print('discard scheduler')

        if self.if_resume:
            checkpoint = torch.load(self.ckpt_path)
            self.Geo_proj.load_state_dict(checkpoint['Geo_proj'])
            self.Cat_proj.load_state_dict(checkpoint['Cat_proj'])
            self.Poi_proj.load_state_dict(checkpoint['Poi_proj'])
            self.Geo_embeds.load_state_dict(checkpoint['Geo_embeds'])
            self.Category_embeds.load_state_dict(checkpoint['Category_embeds'])
            self.Poi_embeds.load_state_dict(checkpoint['Poi_embeds'])
            self.User_embeds.load_state_dict(checkpoint['User_embeds'])
            self.NeuralODE.load_state_dict(checkpoint['NeuralODE'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch_info' in checkpoint.keys():
                self.epoch_info = checkpoint['epoch_info']
            # self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.start_epoch = checkpoint['epoch'] + 1
            self.LOGGER.info(f'resumed from Normal epoch{self.start_epoch - 1}')


    def seed_torch(self, seed):  # 设置种子
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def load_pretrain_fuse(self):
        pretrain_ckpt = torch.load(self.pretrain_fuse_path)
        self.Geo_proj.load_state_dict(pretrain_ckpt['Geo_proj'])
        self.Cat_proj.load_state_dict(pretrain_ckpt['Cat_proj'])
        self.Poi_proj.load_state_dict(pretrain_ckpt['Poi_proj'])
        self.Geo_embeds.load_state_dict(pretrain_ckpt['Geo_embeds'])
        self.Category_embeds.load_state_dict(pretrain_ckpt['Category_embeds'])
        self.Poi_embeds.load_state_dict(pretrain_ckpt['Poi_embeds'])
        self.User_embeds.load_state_dict(pretrain_ckpt['User_embeds'])
        self.NeuralODE.load_state_dict(pretrain_ckpt['NeuralODE'])

    def load_pretrain_truth(self):
        pretrain_ckpt = torch.load(self.pretrain_truth_path)
        self.Geo_proj.load_state_dict(pretrain_ckpt['Geo_proj'])
        self.Cat_proj.load_state_dict(pretrain_ckpt['Cat_proj'])
        self.Poi_proj.load_state_dict(pretrain_ckpt['Poi_proj'])
        self.Geo_embeds.load_state_dict(pretrain_ckpt['Geo_embeds'])
        self.Category_embeds.load_state_dict(pretrain_ckpt['Category_embeds'])
        self.Poi_embeds.load_state_dict(pretrain_ckpt['Poi_embeds'])
        self.User_embeds.load_state_dict(pretrain_ckpt['User_embeds'])
        self.NeuralODE.load_state_dict(pretrain_ckpt['NeuralODE'])

    def load_data(self):
        with open(f'data/{self.data}/processed/param.pkl', 'rb') as f:
            Latsup = pickle.load(f)
            Latinf = pickle.load(f)
            Lonsup = pickle.load(f)
            Loninf = pickle.load(f)
            Ysup = pickle.load(f)
            Yinf = pickle.load(f)
            Msup = pickle.load(f)
            Minf = pickle.load(f)
            Wsup = pickle.load(f)
            Winf = pickle.load(f)
            Dsup = pickle.load(f)
            Dinf = pickle.load(f)
            Hsup = pickle.load(f)
            Hinf = pickle.load(f)
            locations = pickle.load(f)
            n_user = pickle.load(f)
            n_poi = pickle.load(f)
            n_cid = pickle.load(f)
            cid_list = pickle.load(f)
        locations = torch.DoubleTensor(locations).to(self.device)
        return Latsup, Latinf, Lonsup, Loninf, Ysup, Yinf, Msup, Minf, Wsup, Winf, Dsup, Dinf, Hsup, Hinf, \
               locations, n_user, n_poi, n_cid, cid_list

    def show_epoch_info(self):
        if len(self.epoch_info) == 0:
            return
        line = 'epoch, recall2, recall20, mrr2, mrr20\n'
        for info in self.epoch_info:
            line += f'{info[0]}\t'
        line += '\n'

        for j in range(len(self.epoch_info[0][1])):  # res的长度
            for info in self.epoch_info:
                res = info[1]
                if j < 14:
                    line += f'{res[j]:.4f}\t'
                else:
                    line += f'{res[j]:.6f}\t'

            line += '\n'
        self.LOGGER.info(line)

    def fit(self):
        best_val_R1, best_val_R20, _, _, _ = self.eval_model()
        # best_val_R1, best_val_R20 = -1, -1
        self.show_epoch_info()
        best_epoch = 0
        # 打印模型参数

        modeList = [self.Cat_proj, self.Geo_proj, self.Poi_proj, self.Poi_embeds, self.Category_embeds,
                    self.Geo_embeds]  # , Seq_encoder
        # num_params = 0
        # for mode in modeList:
        #     for name in mode.state_dict():
        #         print(name)
        #     for param in mode.parameters():
        #         num_params += param.numel()
        # print('num of params', num_params)

        # criterion = nn.BCEWithLogitsLoss()  # loss函数
        CE = nn.CrossEntropyLoss()
        MSE = nn.MSELoss()
        KLD = nn.KLDivLoss()
        # train_loader = DataLoader(self.train_set_all, self.batch, shuffle=True)
        results = []
        losses = []

        for epoch in range(self.start_epoch, self.epoch + 1):
            self.Geo_proj.train()
            self.Cat_proj.train()
            self.Poi_proj.train()
            self.NeuralODE.train()
            loss_list = []
            train_loader = DataLoader(self.train_set, self.batch, shuffle=True)
            print(f'train on epoch:{epoch}')
            for bn, trn_batch in enumerate(tqdm(train_loader)):
                self.optimizer.zero_grad()
                # with torch.cuda.amp.autocast(dtype=torch.float16):
                trn_batch = trn_batch.to(self.device)
                label = trn_batch.tar_poi

                geo_embeds = self.Geo_embeds(
                    torch.cat([self.locations.float(),
                               torch.zeros(self.n_user, 2).to(self.device)], dim=0)).unsqueeze(-1)
                geo_embeds[self.n_poi + trn_batch.uid] = self.Geo_embeds(
                    self.locations.float()[trn_batch.cur_poi]).unsqueeze(-1)
                poi_embeds = self.Poi_embeds(torch.arange(self.n_poi + self.n_user).to(self.device)).unsqueeze(-1)
                cat_embeds = self.Category_embeds(
                    torch.cat([torch.LongTensor(self.cid_list).to(self.device),
                               torch.arange(self.n_cid, self.n_cid + self.n_user).to(self.device)],
                              dim=0)).unsqueeze(-1)
                multi_embeds = torch.cat([self.Poi_proj(poi_embeds).unsqueeze(1),
                                          self.Cat_proj(cat_embeds).unsqueeze(1),
                                          self.Geo_proj(geo_embeds).unsqueeze(1)], dim=1)

                z = self.NeuralODE(trn_batch, multi_embeds, return_whole_sequence=False)
                A = z[:, :self.NeuralODE.func.n_dim_A]
                x = z[:, self.NeuralODE.func.n_dim_A:self.NeuralODE.func.n_dim_A + self.NeuralODE.func.n_dim_x]
                u = z[:, self.NeuralODE.func.n_dim_A + self.NeuralODE.func.n_dim_x:]
                A = A.reshape(-1, 3, self.graph_size, self.graph_size)
                x = x.reshape(-1, 3, self.graph_size, self.embed)
                u = u.reshape(-1, 3, self.embed)

                truth_preds = torch.einsum('bce, nce-> bcn', multi_embeds[label], multi_embeds[:self.n_poi]).sum(1)
                preds = torch.einsum('bce, nce-> bcn', x[:, :, -1, :], multi_embeds[:self.n_poi]).sum(1) + \
                        torch.einsum('bce, nce-> bcn', u, multi_embeds[:self.n_poi]).sum(1)
                truth_celoss = CE(truth_preds, label)
                u_mseloss = MSE(u.float(), multi_embeds[trn_batch.tar_poi].float())
                # Ax_mseloss = MSE(Ax.float()[:, :, -1, :], multi_embeds[trn_batch.tar_poi].float())
                mseloss = MSE(x.float(), multi_embeds[trn_batch.label_seq].transpose(1, 2).float())

                # KLDloss = KLD(x.float(), multi_embeds[trn_batch.label_seq].transpose(1, 2).float())
                # u_KLDloss = KLD(u, multi_embeds[trn_batch.tar_poi].float())
                celoss = CE(preds, label)
                loss = self.loss_weight*truth_celoss + mseloss + u_mseloss
                # loss = celoss
                # print(KLDloss)
                # print(truth_celoss)
                # print(mseloss)
                # # print(u_mseloss)
                # # print(Ax_mseloss)
                # print(celoss)
                # print(mseloss)
                loss_list.append([truth_celoss.item(), mseloss.item(),u_mseloss.item()])

                # loss.backward(retain_graph=True)
                # self.optimizer.step()
                self.scaler.scale(loss).backward(retain_graph=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.kwargs['is_scheduler']:
                self.scheduler.step()
            self.LOGGER.info(f''' 
            lr = {self.optimizer.state_dict()['param_groups'][0]['lr']}
            Loss : {loss.item()}  = 
            mse: {mseloss.item()} + 
            u_mse: {u_mseloss.item()} +
            truth_celoss : {truth_celoss.item()} +
            ''')
            # 跑完就存
            checkpoint = {
                "Geo_proj": self.Geo_proj.state_dict(),
                "Poi_proj": self.Poi_proj.state_dict(),
                "Cat_proj": self.Cat_proj.state_dict(),
                "Geo_embeds": self.Geo_embeds.state_dict(),
                "Category_embeds": self.Category_embeds.state_dict(),
                "Poi_embeds": self.Poi_embeds.state_dict(),
                "User_embeds": self.User_embeds.state_dict(),
                "NeuralODE": self.NeuralODE.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, f'./ckpts/ckpt.pth')
            self.LOGGER.info(f'normal save at epoch:{epoch}')

            recall_1, recall_20, mrr_1, mrr_20, res = self.eval_model(ifvalid=True)
            self.LOGGER.info('')
            self.LOGGER.info(f'''Epoch: {epoch} / {self.epoch},
            Recall@1: {recall_1}, Recall@20: {recall_20}
            MRR@1: {mrr_1}, MRR@20: {mrr_20}
            ''')

            results.append([truth_celoss.item(), mseloss.item(), u_mseloss.item(), celoss.item(), recall_1.item(), recall_20.item(), mrr_1.item(), mrr_20.item()])
            losses.append([mseloss.item(), u_mseloss.item(), truth_celoss.item(), celoss.item()])
            # with open('results/tky.csv', mode='w', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',')
            #     for row in results:
            #         csv_writer.writerow(row)
            # with open('results/tky_losses.csv', mode='w', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',')
            #     for row in losses:
            #         csv_writer.writerow(row)
            loss_list = np.array(loss_list)
            loss_list = [np.mean(loss_list[:, i]) for i in range(3)]
            assert len(loss_list) == 3
            if epoch % self.info_step == 0:

                self.epoch_info.append([epoch, res + loss_list])
                print('update_info')
                # self.show_epoch_info()

            if recall_1 + recall_20 > best_val_R1 + best_val_R20:
                best_val_R1 = recall_1
                best_val_R20 = recall_20
                best_val_mrr1 = mrr_1
                best_val_mrr20 = mrr_20
                best_epoch = epoch
                self.best_res = res + loss_list
                # best_test_R1, best_test_R20, best_test_mrr1, best_test_mrr20 = self.eval_model(ifvalid=False)
                checkpoint = {
                    "Geo_proj": self.Geo_proj.state_dict(),
                    "Poi_proj": self.Poi_proj.state_dict(),
                    "Cat_proj": self.Cat_proj.state_dict(),
                    "Geo_embeds": self.Geo_embeds.state_dict(),
                    "Category_embeds": self.Category_embeds.state_dict(),
                    "Poi_embeds": self.Poi_embeds.state_dict(),
                    "User_embeds": self.User_embeds.state_dict(),
                    "NeuralODE": self.NeuralODE.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, f'./ckpts/best_new_model.pth')
            if len(self.best_res) != 0:

                self.LOGGER.info(f'''Best Valid at epoch:{best_epoch}''')
                line = ''
                res_key = ['recall_1', 'recall_2', 'recall_3', 'recall_5', 'recall_10', 'recall_15', 'recall_20',
                           'mrr_1', 'mrr_2', 'mrr_3', 'mrr_5', 'mrr_10', 'mrr_15', 'mrr_20', 'celoss', 'poi_mse',
                           'u_mse']
                for key in res_key:
                    line += f'{key}\t'
                line += '\n'
                for i in range(len(self.best_res)):
                    if i < 14:
                        line += f'{self.best_res[i]:.4f}\t'
                    else:
                        line += f'{self.best_res[i]:.6f}\t'

                # for res in self.best_res:
                #     line += f'{res:.4f}\t'
                self.LOGGER.info(line)
            # if epoch - best_epoch == self.patience:
            #     self.LOGGER.info(f'Stop training after {self.patience} epochs without valid improvement.')
            #     break

    def eval_model(self, ifvalid=True):
        preds, truths, labels, cat_labels, user_labels, user_cat_labels = [], [], [], [], [], []
        # MSE = nn.MSELoss(reduce=False)
        with torch.no_grad():
            self.Geo_proj.eval()
            self.Poi_proj.eval()
            self.Cat_proj.eval()
            self.NeuralODE.eval()
            if ifvalid:
                # dataset = self.val_set
                dataset = self.test_set
                # dataset = self.train_set
            else:
                dataset = self.test_set
                # dataset = self.train_set
            loader = DataLoader(dataset, self.batch, shuffle=False)

            for bn, batch in enumerate(tqdm(loader)):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    batch = batch.to(self.device)
                    label = batch.tar_poi

                    geo_embeds = self.Geo_embeds(
                        torch.cat([self.locations.float(),
                                   torch.zeros(self.n_user, 2).to(self.device)], dim=0)).unsqueeze(-1)
                    geo_embeds[self.n_poi + batch.uid] = self.Geo_embeds(
                        self.locations.float()[batch.cur_poi]).unsqueeze(-1)
                    poi_embeds = self.Poi_embeds(torch.arange(self.n_poi + self.n_user).to(self.device)).unsqueeze(-1)
                    cat_embeds = self.Category_embeds(
                        torch.cat([torch.LongTensor(self.cid_list).to(self.device),
                                   torch.arange(self.n_cid, self.n_cid + self.n_user).to(self.device)],
                                  dim=0)).unsqueeze(-1)
                    multi_embeds = torch.cat([self.Poi_proj(poi_embeds).unsqueeze(1),
                                              self.Cat_proj(cat_embeds).unsqueeze(1),
                                              self.Geo_proj(geo_embeds).unsqueeze(1)], dim=1)

                    z = self.NeuralODE(batch, multi_embeds, return_whole_sequence=False)
                    A = z[:, :self.NeuralODE.func.n_dim_A]
                    x = z[:, self.NeuralODE.func.n_dim_A:self.NeuralODE.func.n_dim_A + self.NeuralODE.func.n_dim_x]
                    u = z[:, self.NeuralODE.func.n_dim_A + self.NeuralODE.func.n_dim_x:]
                    A = A.reshape(-1, 3, self.graph_size, self.graph_size)
                    x = x.reshape(-1, 3, self.graph_size, self.embed)
                    u = u.reshape(-1, 3, self.embed)
                    # Ax = A @ x
                    # Ax = torch.diff(Ax, dim=-2, prepend=torch.zeros_like(Ax[:, :, 0, :]).unsqueeze(2))
                    pred = x[:, :, -1, :]

                    pred = torch.einsum('bce, nce-> bcn', pred, multi_embeds[:self.n_poi]).sum(1)
                    pred += torch.einsum('bce, nce-> bcn', u, multi_embeds[:self.n_poi]).sum(1)
                    # pred += torch.einsum('bce, nce-> bcn', Ax[:, :, -1, :], multi_embeds[:self.n_poi]).sum(1)
                    assert not torch.isnan(pred).any()
                    # print(pred)

                assert list(pred.shape) == [len(batch), self.n_poi]

                logit = pred
                logits = logit \
                    .squeeze().clone().detach().cpu().numpy()

                label = label.squeeze().clone().detach().cpu().numpy()
                labels.append(label)
                preds.append(logits)

        preds = torch.FloatTensor(np.concatenate(preds, axis=0)).to(self.device)
        labels = torch.LongTensor(np.concatenate(labels, axis=0)).to(self.device)

        recall_1 = hit_rate(input=preds, target=labels, k=1)
        recall_2 = hit_rate(input=preds, target=labels, k=2)
        recall_3 = hit_rate(input=preds, target=labels, k=3)
        recall_5 = hit_rate(input=preds, target=labels, k=5)
        recall_10 = hit_rate(input=preds, target=labels, k=10)
        recall_15 = hit_rate(input=preds, target=labels, k=15)
        recall_20 = hit_rate(input=preds, target=labels, k=20)
        rr_1 = reciprocal_rank(input=preds, target=labels, k=1)
        rr_2 = reciprocal_rank(input=preds, target=labels, k=2)
        rr_3 = reciprocal_rank(input=preds, target=labels, k=3)
        rr_5 = reciprocal_rank(input=preds, target=labels, k=5)
        rr_10 = reciprocal_rank(input=preds, target=labels, k=10)
        rr_15 = reciprocal_rank(input=preds, target=labels, k=15)
        rr_20 = reciprocal_rank(input=preds, target=labels, k=20)

        recall_1 = recall_1.sum() / recall_1.shape[0] * 100
        recall_2 = recall_2.sum() / recall_2.shape[0] * 100
        recall_3 = recall_3.sum() / recall_3.shape[0] * 100
        recall_5 = recall_5.sum() / recall_5.shape[0] * 100
        recall_10 = recall_10.sum() / recall_10.shape[0] * 100
        recall_15 = recall_15.sum() / recall_15.shape[0] * 100
        recall_20 = recall_20.sum() / recall_20.shape[0] * 100
        mrr_1 = rr_1.mean() * 100
        mrr_2 = rr_2.mean() * 100
        mrr_3 = rr_3.mean() * 100
        mrr_5 = rr_5.mean() * 100
        mrr_10 = rr_10.mean() * 100
        mrr_15 = rr_15.mean() * 100
        mrr_20 = rr_20.mean() * 100


        result = [recall_1, recall_2, recall_3, recall_5, recall_10, recall_15, recall_20, mrr_1, mrr_2, mrr_3, mrr_5,
                  mrr_10, mrr_15, mrr_20]
        res_key = ['recall_1', 'recall_2', 'recall_3', 'recall_5', 'recall_10', 'recall_15', 'recall_20',
                   'mrr_1', 'mrr_2', 'mrr_3', 'mrr_5', 'mrr_10', 'mrr_15', 'mrr_20']
        line = ''
        for key in res_key:
            line += f'{key}\t'
        line += '\n'
        for res in result:
            line += f'{res:.4f}\t'
        self.LOGGER.info(line)
        return recall_1, recall_20, mrr_1, mrr_20, result
