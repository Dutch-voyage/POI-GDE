import pickle
import torch
import tqdm
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data  # Data类？
from torch_geometric.utils import k_hop_subgraph, to_undirected, subgraph


class MultiSessionsGraph(InMemoryDataset):
    def __init__(self, device, window_size, root='../processed/nyc', phrase='train', transform=None, pre_transform=None):
        self.phrase = phrase
        self.device = device
        self.window_size = window_size
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.pkl', '../lt_dist_graph.pkl']

    @property
    def processed_file_names(self):
        return [self.phrase + '_session_graph_' + '.pt']

    def download(self):
        pass

    def process(self):
        # print(self.raw_dir + '/' + self.raw_file_names[0])
        with open(self.raw_dir + '/' + self.raw_file_names[0], 'rb') as f:  # train.pkl
            data = pickle.load(f)
        data_list = []
        print(self.raw_dir + '/' + self.raw_file_names[0])
        for uid, input_seq, label_seq, input_cidseq, label_cidseq, input_timeseq, label_timeseq, tar_poi, tar_cid, tar_time, y in tqdm(data):
            i, input_x, input_cid, input_senders, input_nodes = 0, [], [], [], {}
            input_time = []
            input_freq = torch.zeros(self.window_size, self.window_size)
            for j, node in enumerate(input_seq):  # 遍历点列表 [当前traj长度]
                if node not in input_nodes:
                    input_nodes[node] = i
                    input_x.append([node])
                    input_cid.append([input_cidseq[j]])  # 记下点的类别
                    i += 1
                input_time.append([input_timeseq[j]])  # 记下点的时间
                input_senders.append(input_nodes[node])  # 相当于对点（在当前序列内）重新编号

            label_time = []
            i, label_x, label_cid, label_senders, label_nodes = 0, [], [], [], {}
            label_freq = torch.zeros(self.window_size, self.window_size)
            for j, node in enumerate(label_seq):  # 遍历点列表 [当前traj长度]
                if node not in label_nodes:
                    label_nodes[node] = i
                    label_x.append([node])
                    label_cid.append([label_cidseq[j]])  # 记下点的类别
                    i += 1
                label_time.append([label_timeseq[j]])
                label_senders.append(label_nodes[node])  # 相当于对点（在当前序列内）重新编号

            input_idx = torch.LongTensor(input_senders)
            for j, node in enumerate(input_seq):
                num = (input_idx == input_nodes[node]).sum()
                idx = (input_idx == input_nodes[node]).nonzero().squeeze(1)
                input_freq[j:, idx] += 1 / num
            input_freq = input_freq.unsqueeze(0)

            label_idx = torch.LongTensor(label_senders)
            for j, node in enumerate(label_seq):
                num = (label_idx == label_nodes[node]).sum()
                idx = (label_idx == label_nodes[node]).nonzero().squeeze(1)
                label_freq[j:, idx] += 1 / num
            label_freq = label_freq.unsqueeze(0)

            uid = torch.LongTensor([uid])  # 样本的uid
            tar_poi = torch.LongTensor([tar_poi])  # 目标poi
            cur_poi = torch.LongTensor([input_seq[len(input_seq) - 1]])
            cur_time = torch.FloatTensor([input_time[len(input_time) - 1]]).squeeze(1)
            tar_time = torch.FloatTensor([tar_time])
            y = torch.LongTensor([y])  # 01标签

            input_edge = [input_senders[: -1], input_senders[1:]]
            input_edge_index = torch.LongTensor([input_edge])
            input_seq = torch.LongTensor([input_seq])

            label_edge = [label_senders[: -1], label_senders[1:]]
            label_edge_index = torch.LongTensor([label_edge])
            label_seq = torch.LongTensor([label_seq])

            input_timeseq = torch.FloatTensor(input_time).squeeze(1).unsqueeze(0)
            label_timeseq = torch.FloatTensor(label_time).squeeze(1).unsqueeze(0)


            data_list.append(Data(input_seq=input_seq, input_edges=input_edge_index,
                                  label_seq=label_seq, label_edges=label_edge_index,
                                  input_freq=input_freq, label_freq=label_freq,
                                  input_timeseq=input_timeseq, label_timeseq=label_timeseq,
                                  uid=uid, tar_poi=tar_poi, cur_poi=cur_poi,
                                  tar_time=tar_time, cur_time=cur_time,
                                  y=y,
                                  ))  # 还记下了样本的经纬度

        data, slices = self.collate(data_list)  # 这个方法有什么用？
        # 会生成3个pt文件
        torch.save((data, slices), self.processed_paths[0])  # 是存数据的方法，注释掉后不会生成pt文件

