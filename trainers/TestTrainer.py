import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

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


from dataloaders.dataloader_ode import MultiSessionsGraph
from Metrics.metrics import hit_rate, reciprocal_rank, SortLoss
from utils.Scheduler import GradualWarmupScheduler


class TestTrainer:
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

        LOG_FORMAT = "%(asctime)s  %(message)s"
        DATE_FORMAT = "%m/%d %H:%M"

        console_handler = logging.StreamHandler()  # 输出到控制台
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        self.LOGGER = logging.getLogger(__name__)
        self.LOGGER.addHandler(console_handler)

        if self.log is not None:
            log_name = self.log + time.strftime(
                '%Y-%m-%d-%H-%M-%S') + '_UACGNN_Full_GeoCom' + f'_{self.data}_ ' + f'{self.if_resume}' + '.log'
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

        self.Poi_embeds = EmbeddingLayer(self.n_poi + self.n_user, self.embed).to(self.device)
        self.Geo_embeds = GeoEmbedding(self.Latsup, self.Latinf, self.Lonsup, self.Loninf, self.embed, self.device).to(self.device)
        self.User_embeds = EmbeddingLayer(self.n_user, self.embed).to(self.device)
        self.Category_embeds = EmbeddingLayer(self.n_cid + self.n_user, self.embed).to(self.device)
        self.Time_embeds = TimeEmbedding(self.Ysup, self.Yinf, self.Msup, self.Minf, self.Wsup, self.Winf,
                                         self.Dsup, self.Dinf, self.Hsup, self.Hinf,
                                         self.embed, self.device).to(self.device)

        self.Poi_proj = MultiFuser(self.embed, 1, 1).to(self.device)
        self.Cat_proj = MultiFuser(self.embed, 1, 1).to(self.device)
        self.Geo_proj = MultiFuser(self.embed, 1, 1).to(self.device)

        self.UserInterest = UserInterest(self.embed).to(self.device)
        self.Scoring = Scoring(self.embed).to(self.device)
        self.UserScoring = Scoring(self.embed).to(self.device)

        self.linear = nn.Linear(self.graph_size, 1).to(self.device)

        self.GraphODEF = GraphODEF(self.Time_embeds, self.graph_size, self.embed, self.device).to(self.device)

        self.NeuralODE = NeuralODE(self.GraphODEF, self.n_poi).to(self.device)

        self.Swish = Swish().to(self.device)

        if self.if_pretrain_fuse:
            self.load_pretrain_fuse()
        if self.if_pretrain_truth:
            self.load_pretrain_truth()

        self.scaler = torch.cuda.amp.GradScaler()

        self.optimizer = torch.optim.Adam([
            {'params': self.linear.parameters()},
            {'params': self.Swish.parameters()},
            {'params': self.UserInterest.parameters()},
            {'params': self.Scoring.parameters()},
            {'params': self.UserScoring.parameters()},
            {'params': self.NeuralODE.parameters()},
            {'params': self.Geo_proj.parameters()},
            {'params': self.Cat_proj.parameters()},
            {'params': self.Poi_proj.parameters()},
            {'params': self.Geo_embeds.parameters()},
            {'params': self.Category_embeds.parameters()},
            {'params': self.User_embeds.parameters()},
            {'params': self.Poi_embeds.parameters()},
        ],
            lr=self.lr, weight_decay=self.weight_decay)
        cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, T_max=self.T_max, eta_min=0, last_epoch=-1)
        warmUpScheduler = GradualWarmupScheduler(optimizer=self.optimizer, multiplier=self.multiplier,
                                                 warm_epoch=self.epoch // 10, after_scheduler=cosineScheduler)
        self.scheduler = warmUpScheduler

        if self.if_resume:
            checkpoint = torch.load(self.ckpt_path)
            self.linear.load_state_dict(checkpoint['linear'])
            self.Swish.load_state_dict(checkpoint['Swish'])
            self.Geo_proj.load_state_dict(checkpoint['Geo_proj'])
            self.Cat_proj.load_state_dict(checkpoint['Cat_proj'])
            self.Poi_proj.load_state_dict(checkpoint['Poi_proj'])
            self.UserInterest.load_state_dict(checkpoint['UserInterest'])
            self.Scoring.load_state_dict(checkpoint['Scoring'])
            self.UserScoring.load_state_dict(checkpoint['UserScoring'])
            self.GraphODEF.load_state_dict(checkpoint['GraphODEF'])
            self.NeuralODE.load_state_dict(checkpoint['NeuralODE'])
            self.Geo_embeds.load_state_dict(checkpoint['Geo_embeds'])
            self.Category_embeds.load_state_dict(checkpoint['Category_embeds'])
            self.Poi_embeds.load_state_dict(checkpoint['Poi_embeds'])
            self.User_embeds.load_state_dict(checkpoint['User_embeds'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.start_epoch = checkpoint['epoch'] + 1
            self.LOGGER.info(f'resumed from epoch{self.start_epoch - 1}')

    def BPR_Loss(self, pos, neg):
        rank = ((neg > pos).sum(dim=-1) + 1).unsqueeze(-1)
        loss = -torch.log(self.Swish((pos - neg), - torch.log(rank / self.n_poi)) + 1e-8)
        return loss.mean()

    def seed_torch(self, seed):  # 设置种子
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def load_pretrain_fuse(self):
        pretrain_ckpt = torch.load(self.pretrain_fuse_path)
        self.linear.load_state_dict(pretrain_ckpt['linear'])
        self.Swish.load_state_dict(pretrain_ckpt['Swish'])
        self.Geo_proj.load_state_dict(pretrain_ckpt['Geo_proj'])
        self.Cat_proj.load_state_dict(pretrain_ckpt['Cat_proj'])
        self.Poi_proj.load_state_dict(pretrain_ckpt['Poi_proj'])
        self.UserInterest.load_state_dict(pretrain_ckpt['UserInterest'])
        self.Scoring.load_state_dict(pretrain_ckpt['Scoring'])
        self.UserScoring.load_state_dict(pretrain_ckpt['UserScoring'])
        self.GraphODEF.load_state_dict(pretrain_ckpt['GraphODEF'])
        self.NeuralODE.load_state_dict(pretrain_ckpt['NeuralODE'])
        self.Geo_embeds.load_state_dict(pretrain_ckpt['Geo_embeds'])
        self.Category_embeds.load_state_dict(pretrain_ckpt['Category_embeds'])
        self.Poi_embeds.load_state_dict(pretrain_ckpt['Poi_embeds'])
        self.User_embeds.load_state_dict(pretrain_ckpt['User_embeds'])

    def load_pretrain_truth(self):
        pretrain_ckpt = torch.load(self.pretrain_truth_path)
        self.linear.load_state_dict(pretrain_ckpt['linear'])
        self.Swish.load_state_dict(pretrain_ckpt['Swish'])
        self.Geo_proj.load_state_dict(pretrain_ckpt['Geo_proj'])
        self.Cat_proj.load_state_dict(pretrain_ckpt['Cat_proj'])
        self.Poi_proj.load_state_dict(pretrain_ckpt['Poi_proj'])
        self.UserInterest.load_state_dict(pretrain_ckpt['UserInterest'])
        self.Scoring.load_state_dict(pretrain_ckpt['Scoring'])
        self.UserScoring.load_state_dict(pretrain_ckpt['UserScoring'])
        self.GraphODEF.load_state_dict(pretrain_ckpt['GraphODEF'])
        self.NeuralODE.load_state_dict(pretrain_ckpt['NeuralODE'])
        self.Geo_embeds.load_state_dict(pretrain_ckpt['Geo_embeds'])
        self.Category_embeds.load_state_dict(pretrain_ckpt['Category_embeds'])
        self.Poi_embeds.load_state_dict(pretrain_ckpt['Poi_embeds'])
        self.User_embeds.load_state_dict(pretrain_ckpt['User_embeds'])

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

    def fit(self):
        # best_val_R1, best_val_R20, _, _ = self.eval_model()
        best_val_R1, best_val_R20 = -1, -1
        best_epoch = 0
        # 打印模型参数

        modeList = [self.NeuralODE, self.Cat_proj, self.Geo_proj, self.Poi_proj, self.Poi_embeds, self.Category_embeds, self.Geo_embeds]  # , Seq_encoder
        num_params = 0
        for mode in modeList:
            for name in mode.state_dict():
                print(name)
            for param in mode.parameters():
                num_params += param.numel()
        print('num of params', num_params)

        # criterion = nn.BCEWithLogitsLoss()  # loss函数
        CE = nn.CrossEntropyLoss()
        MSE = nn.MSELoss()
        KL = nn.KLDivLoss()

        # train_loader = DataLoader(self.train_set_all, self.batch, shuffle=True)

        for epoch in range(self.start_epoch, self.epoch + 1):
            self.Time_embeds.train()
            self.GraphODEF.train()
            self.NeuralODE.train()
            self.UserInterest.train()
            self.Scoring.train()
            self.Geo_proj.train()
            self.Cat_proj.train()
            self.Poi_proj.train()
            '''
            if self.if_pretrain_fuse:
                for param in self.Poi_embeds.parameters():
                    param.requires_grad = False
                for param in self.User_embeds.parameters():
                    param.requires_grad = False
                for param in self.Category_embeds.parameters():
                    param.requires_grad = False
                for param in self.Geo_embeds.parameters():
                    param.requires_grad = False
                for param in self.Poi_proj.parameters():
                    param.requires_grad = False
                for param in self.Cat_proj.parameters():
                    param.requires_grad = False
                for param in self.Geo_proj.parameters():
                    param.requires_grad = False
            '''
            train_loader = DataLoader(self.train_set, self.batch, shuffle=True)
            for bn, trn_batch in enumerate(tqdm(train_loader)):
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    trn_batch = trn_batch.to(self.device)
                    label = torch.cat([trn_batch.input_seq, trn_batch.label_seq], dim=-1)

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

                    # fuse_embeds = self.MultiFuser2([cat_embeds, geo_embeds])
                    '''
                    user_embeds = self.User_embeds(torch.arange(self.n_user).to(self.device)).unsqueeze(2)

                    geo_embeds = self.Geo_embeds(self.locations.float()).unsqueeze(2)
                    poi_embeds = self.Poi_embeds(torch.arange(self.n_poi).to(self.device)).unsqueeze(2)
                    cat_embeds = self.Category_embeds(torch.LongTensor(self.cid_list).to(self.device)).unsqueeze(2)

                    multi_embeds = torch.cat([self.Poi_proj(poi_embeds).unsqueeze(1),
                                              self.Cat_proj(cat_embeds).unsqueeze(1),
                                              self.Geo_proj(geo_embeds).unsqueeze(1)], dim=1)
                    multi_usr_embeds = torch.cat([self.Poi_proj(user_embeds).unsqueeze(1),
                                                  self.Cat_proj(user_embeds).unsqueeze(1),
                                                  self.Geo_proj(user_embeds).unsqueeze(1)], dim=1)
                    multi_embeds = torch.cat([multi_embeds, multi_usr_embeds], dim=0)
                    '''
                    x = self.NeuralODE(trn_batch, multi_embeds, return_whole_sequence=True)
                    # assert not torch.isnan(x).any()
                    At, xt, ut, x_shape = getAxu(multi_embeds, self.n_poi, trn_batch.label_seq, trn_batch.label_edges,
                                                 trn_batch.uid)

                    tA = self.GraphODEF.time_embedding(trn_batch.label_timeseq).unsqueeze(1).repeat(1, 3, 1, 1)

                    steps, bs, seqlen, _ = x.shape
                    x = x.reshape(steps, bs, seqlen, *x_shape).transpose(2, 3)

                    pred = x
                    pred = [pred[i, :, :, - (i + 1), :].unsqueeze(0) for i in range(steps - 1)]
                    pred = torch.cat(pred, dim=0).mean(0)
                    # pred = pred[0, :, :, -1, :]

                    assert not torch.isnan(pred).any()
                    """
                    for i in range(steps):
                        for j in range(steps):
                            pred = x
                            pred = [pred[i, :, :, - (i + 1), :].unsqueeze(0) for i in range(steps - 1)]
                            pred = torch.cat(pred, dim=0).mean(0).nan_to_num()

                            assert not torch.isnan(pred).any()
                            '''
                            uA = u.reshape(seqlen, bs, seqlen, *x_shape).transpose(2, 3)
                            uA = [uA[i, :, :, - (i + 1), :].unsqueeze(0) for i in range(steps - 1)]
                            uA = torch.cat(uA, dim=0).permute(1, 2, 0, 3)
                            assert not torch.isnan(uA).any()

                            ui = self.UserInterest(uA, tA[:, :, 1:, :], multi_embeds[:self.n_poi])
                            assert not torch.isnan(ui).any()
                            '''
                            preds = torch.einsum('bce, nce-> bcn', pred, multi_embeds[:self.n_poi])
                            # 
                            preds = preds.sum(1).nan_to_num()
                            assert not torch.isnan(preds).any()
                            cri_losses += CE(preds, label[:, i + j + 1])
                    """
                    # preds = torch.einsum('bce, nce-> bcn', multi_embeds[trn_batch.tar_poi], multi_embeds[:self.n_poi]).sum(1)
                    preds = torch.einsum('bce, nce-> bcn', pred, multi_embeds[:self.n_poi]).sum(1)
                loss = CE(preds, label[:, 16])
                # loss = cri_losses / (steps * steps)
                print(loss)

                # loss.backward(retain_graph=True)
                # self.optimizer.step()
                self.scaler.scale(loss).backward(retain_graph=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.scheduler.step()
            self.LOGGER.info(f''' 
            lr = {self.optimizer.state_dict()['param_groups'][0]['lr']}
            Loss : {loss.item()} =
            cri_loss : {loss.item()}
            ''')
            recall_1, recall_20, mrr_1, mrr_20 = self.eval_model(ifvalid=True)
            self.LOGGER.info('')
            self.LOGGER.info(f'''Epoch: {epoch} / {self.epoch},
            Recall@5: {recall_1}, Recall@20: {recall_20}
            MRR@5: {mrr_1}, MRR@20: {mrr_20}
            ''')

            if recall_1 > best_val_R1 or recall_20 > best_val_R20:
                best_val_R1 = recall_1
                best_val_R20 = recall_20
                best_val_mrr1 = mrr_1
                best_val_mrr20 = mrr_20
                best_epoch = epoch
                best_test_R1, best_test_R20, best_test_mrr1, best_test_mrr20 = self.eval_model(ifvalid=False)
                checkpoint = {
                    "linear": self.linear.state_dict(),
                    "Swish": self.Swish.state_dict(),
                    "UserInterest": self.UserInterest.state_dict(),
                    "Scoring": self.Scoring.state_dict(),
                    "UserScoring": self.UserScoring.state_dict(),
                    "GraphODEF": self.GraphODEF.state_dict(),
                    "NeuralODE": self.NeuralODE.state_dict(),
                    "Geo_proj": self.Geo_proj.state_dict(),
                    "Poi_proj": self.Poi_proj.state_dict(),
                    "Cat_proj": self.Cat_proj.state_dict(),
                    "Geo_embeds": self.Geo_embeds.state_dict(),
                    "Category_embeds": self.Category_embeds.state_dict(),
                    "Poi_embeds": self.Poi_embeds.state_dict(),
                    "User_embeds": self.User_embeds.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, f'./ckpts/best_G2G_model.pth')
            if best_epoch != 0:
                self.LOGGER.info(f'''Best Valid R@1: {best_val_R1} R@20: {best_val_R20} at epoch {best_epoch}, 
                                                MRR@1: {best_val_mrr1} MRR@20: {best_val_mrr20}, 
                                     Best Test R@1: {best_test_R1} R@20: {best_test_R20} at epoch {best_epoch},
                                                MRR@1: {best_test_mrr1} MRR@20: {best_test_mrr20}\n
            \n''')
            if epoch - best_epoch == self.patience:
                self.LOGGER.info(f'Stop training after {self.patience} epochs without valid improvement.')
                break

    def eval_model(self, ifvalid=True):
        preds, truths, labels, cat_labels, user_labels, user_cat_labels = [], [], [], [], [], []
        # MSE = nn.MSELoss(reduce=False)
        with torch.no_grad():
            self.GraphODEF.eval()
            self.NeuralODE.eval()
            self.UserInterest.eval()
            self.Scoring.eval()
            self.Geo_proj.eval()
            self.Poi_proj.eval()
            self.Cat_proj.eval()
            if ifvalid:
                # dataset = self.val_set
                dataset = self.test_set
            else:
                dataset = self.test_set
            loader = DataLoader(dataset, self.batch, shuffle=True)

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

                    # fuse_embeds = self.MultiFuser2([cat_embeds, geo_embeds])
                    '''
                    user_embeds = self.User_embeds(torch.arange(self.n_user).to(self.device)).unsqueeze(2)

                    geo_embeds = self.Geo_embeds(self.locations.float()).unsqueeze(2)
                    poi_embeds = self.Poi_embeds(torch.arange(self.n_poi).to(self.device)).unsqueeze(2)
                    cat_embeds = self.Category_embeds(torch.LongTensor(self.cid_list).to(self.device)).unsqueeze(2)

                    multi_embeds = torch.cat([self.Poi_proj(poi_embeds).unsqueeze(1),
                                              self.Cat_proj(cat_embeds).unsqueeze(1),
                                              self.Geo_proj(geo_embeds).unsqueeze(1)], dim=1)
                    multi_usr_embeds = torch.cat([self.Poi_proj(user_embeds).unsqueeze(1),
                                                  self.Cat_proj(user_embeds).unsqueeze(1),
                                                  self.Geo_proj(user_embeds).unsqueeze(1)], dim=1)
                    multi_embeds = torch.cat([multi_embeds, multi_usr_embeds], dim=0)
                    '''
                    x = self.NeuralODE(batch, multi_embeds, return_whole_sequence=True)
                    assert not torch.isnan(x).any()

                    At, xt, ut, x_shape = getAxu(multi_embeds, self.n_poi, batch.label_seq, batch.label_edges,
                                                 batch.uid)

                    tA = self.GraphODEF.time_embedding(batch.label_timeseq).unsqueeze(1).repeat(1, 3, 1, 1)

                    steps, bs, seqlen, _ = x.shape
                    x = x.reshape(steps, bs, seqlen, *x_shape).transpose(2, 3)

                    pred = x
                    pred = [pred[i, :, :, - (i + 1), :].unsqueeze(0) for i in range(steps - 1)]
                    pred = torch.cat(pred, dim=0).mean(0)
                    # pred = pred[0, :, :, -1, :]

                    assert not torch.isnan(pred).any()
                    '''
                    uA = u.reshape(seqlen, bs, seqlen, *x_shape).transpose(2, 3)
                    uA = [uA[i, :, :, - (i + 1), :].unsqueeze(0) for i in range(steps - 1)]
                    uA = torch.cat(uA, dim=0).permute(1, 2, 0, 3)
                    assert not torch.isnan(uA).any()

                    ui = self.UserInterest(uA, tA[:, :, 1:, :], multi_embeds[:self.n_poi])
                    assert not torch.isnan(ui).any()
                    '''
                    pred = torch.einsum('bce, nce-> bcn', pred, multi_embeds[:self.n_poi])
                    # pred = torch.einsum('bce, nce-> bcn', multi_embeds[batch.tar_poi], multi_embeds[:self.n_poi])
                    pred = pred.sum(1).nan_to_num()
                    assert not torch.isnan(pred).any()
                print(pred)

                assert list(pred.shape) == [len(batch), self.n_poi]

                logit = pred
                logits = logit \
                    .squeeze().clone().detach().cpu().numpy()

                label = label.squeeze().clone().detach().cpu().numpy()
                labels.append(label)
                # cat_label = cat_label.squeeze().clone().detach().cpu().numpy()
                # cat_labels.append(cat_label)

                '''
                recall = hit_rate(input=preds, target=label, k=1)
                recall = recall.sum() / recall.shape[0]

                self.LOGGER.info(f'batch{bn} Recall@1: {recall}')
                '''
                preds.append(logits)

        preds = torch.FloatTensor(np.concatenate(preds, axis=0)).to(self.device)
        labels = torch.LongTensor(np.concatenate(labels, axis=0)).to(self.device)

        recall_1 = hit_rate(input=preds, target=labels, k=1)
        recall_20 = hit_rate(input=preds, target=labels, k=20)
        rr_1 = reciprocal_rank(input=preds, target=labels, k=1)
        rr_20 = reciprocal_rank(input=preds, target=labels, k=20)

        recall_1 = recall_1.sum() / recall_1.shape[0]
        recall_20 = recall_20.sum() / recall_20.shape[0]
        mrr_1 = rr_1.mean()
        mrr_20 = rr_20.mean()

        print(recall_1, recall_20, mrr_1, mrr_20)
        '''
        recalls = [0]
        for i in tqdm(range(1, self.n_poi)):

            recall = hit_rate(input=com_preds, target=labels, k=i)
            recall = recall.sum() / recall.shape[0]
            recalls.append(recall.clone().detach().cpu().numpy())
            if i == self.n_poi // (2 ** (epoch % 16)):
                print(f'recall@{i}: {recall}')
        recalls = np.array(recalls)
        recalls = np.diff(recalls)
        plt.plot(range(1, self.n_poi), recalls)
        plt.savefig(f'figures/recalls_{epoch}.png')
        '''

        return recall_1, recall_20, mrr_1, mrr_20
