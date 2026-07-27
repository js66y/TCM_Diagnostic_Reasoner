#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong
import numpy as np
# import pandas as pd
# import random
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch import Tensor
import numpy as np
import torch
from torch_geometric.data import Data
# from torch_sparse import SparseTensor


class OneHotEncoding0d(nn.Module):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)  # 7
        return torch.cat(
            [
                nn.functional.one_hot(x[..., i], cardinality)
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )

sh_edge = np.load('C:/Users/711/Downloads/KDHR-main/KDHR-main/data/sh_graph.npy')
sh_edge = sh_edge.tolist()
sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
sh_x = torch.tensor([[i] for i in range(1195)], dtype=torch.float)
sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous()).to("cpu")
# sh_data_adj = SparseTensor(row=sh_data.edge_index[0], col=sh_data.edge_index[1],
#                            sparse_sizes=(1195, 1195))
# S-S G
ss_edge = np.load('C:/Users/711/Downloads/KDHR-main/KDHR-main/data/ss_graph.npy')
ss_edge = ss_edge.tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
ss_x = torch.tensor([[i] for i in range(390)], dtype=torch.float)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous()).to("cpu")
# ss_data_adj = SparseTensor(row=ss_data.edge_index[0], col=ss_data.edge_index[1],
#                            sparse_sizes=(390, 390))

# H-H G
hh_edge = np.load('C:/Users/711/Downloads/KDHR-main/KDHR-main/data/hh_graph.npy').tolist()
hh_edge_index = torch.tensor(hh_edge, dtype=torch.long) - 390  # 边索引需要减去390
hh_x = torch.tensor([[i] for i in range(390, 1195)], dtype=torch.float)
hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous()).to("cpu")
# hh_data_adj = SparseTensor(row=hh_data.edge_index[0], col=hh_data.edge_index[1],
#                            sparse_sizes=(805, 805))


seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

class GCNConv_SH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SH, self).__init__(aggr='mean')  # 对邻居节点特征进行平均操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # 公式2
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)  # m = e*T 公式1
        return x_j

class GCNConv_SS_HH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SS_HH, self).__init__(aggr='add')  # 对邻居节点特征进行sum操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # 公式10
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)
        return x_j

class TCM_Model(torch.nn.Module):
    def __init__(self, dropout, emb_dim, num_herb, cat_cardinalities, device):
    # def __init__(self, ss_num, hh_num, sh_num, embedding_dim=64, batchSize=64, drop=0.0):
        super(TCM_Model, self).__init__()
        self.device = device
        self.cat_module = OneHotEncoding0d(cat_cardinalities)
        ss_num = 390
        hh_num = 805
        sh_num = 1195
        embedding_dim = 64
        batchSize = 64
        drop = 0
        self.device = device
        self.batchSize = batchSize
        self.dropout = drop
        self.SH_embedding = torch.nn.Embedding(sh_num, embedding_dim)
        # S-H 图所需的网络
        # S
        self.convSH_TostudyS_1 = GCNConv_SH(embedding_dim, embedding_dim)

        self.convSH_TostudyS_2 = GCNConv_SH(embedding_dim, embedding_dim)

        # self.convSH_TostudyS_3 = GCNConv_SH(embedding_dim, embedding_dim)

        self.SH_mlp_1 = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1 = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1 = torch.nn.Tanh()
        # H
        self.convSH_TostudyS_1_h = GCNConv_SH(embedding_dim, embedding_dim)

        self.convSH_TostudyS_2_h = GCNConv_SH(embedding_dim, embedding_dim)

        # self.convSH_TostudyS_3_h = GCNConv_SH(embedding_dim, embedding_dim)

        self.SH_mlp_1_h = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1_h = torch.nn.Tanh()
        # S-S图网络
        self.convSS = GCNConv_SS_HH(embedding_dim, 256)
        # H-H图网络  维度加上嵌入KG特征的维度
        self.convHH = GCNConv_SS_HH(embedding_dim, 256)
        # self.convHH = GCNConv_SS_HH(embedding_dim, 256)
        # SI诱导层
        # SUM
        self.mlp = torch.nn.Linear(256, 256)
        # cat
        # self.mlp = torch.nn.Linear(512, 512)
        self.SI_bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.up2pre = torch.nn.Linear(256, 1350)
        self.down2pre = torch.nn.Linear(390, 256)
        self.class_head1 = nn.Linear(805, 512)
        self.regression_head1 = nn.Linear(805, 512)
        self.class_head2 = nn.Linear(512, num_herb)
        self.regression_head2 = nn.Linear(512, num_herb)

    def forward(self, X, bank, gat):
        x = []
        x.append(X[0])
        x.append(self.cat_module(X[1][:,:4]).float())
        symptom_OH = torch.column_stack([x_.flatten(1, -1) for x_ in x])
        one_hot = torch.zeros(X[1].shape[0], 174, dtype=torch.float32, device=self.device)
        one_hot.scatter_(1, X[1][:, 4:].long(), 1)                #             1   2   3
        OH = torch.concat((symptom_OH, one_hot), dim=1) # 64, 1350 = 1+2+654+207+312+174
        x_SH, edge_index_SH, x_SS, edge_index_SS, x_HH, edge_index_HH = sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index, hh_data.x, hh_data.edge_index, 
    # def forward(self, x_SH, edge_index_SH, x_SS, edge_index_SS, x_HH, edge_index_HH, prescription, kgOneHot=None):
        # [1195, 1]、[2, 79870]、[390, 1]、[2, 2546]、[805, 1]、[2, 9038]、[512, 390]、[805, 27]
        # S-H图搭建
        # 第一层
        x_SH1 = self.SH_embedding(x_SH.long())
        x_SH1 = x_SH1.squeeze(1)
        x_SH2 = self.convSH_TostudyS_1(x_SH1.float(), edge_index_SH)
        # 第二层
        x_SH6 = self.convSH_TostudyS_2(x_SH2, edge_index_SH)
        # x_SH6 = x_SH6.view(-1, 256)
        # 第三层
        # x_SH7 = self.convSH_TostudyS_3(x_SH6, edge_index_SH)

        x_SH9 = (x_SH1 + x_SH2 + x_SH6 ) / 3.0
        x_SH9 = self.SH_mlp_1(x_SH9)
        x_SH9 = x_SH9.view(1195, -1)
        x_SH9 = self.SH_bn_1(x_SH9)
        x_SH9 = self.SH_tanh_1(x_SH9)
        # SH H
        x_SH11 = self.SH_embedding(x_SH.long())
        x_SH11 = x_SH11.squeeze(1)
        x_SH22 = self.convSH_TostudyS_1_h(x_SH11.float(), edge_index_SH)
        # 第二层
        x_SH66 = self.convSH_TostudyS_2_h(x_SH22, edge_index_SH)
        # x_SH66 = x_SH66.view(-1, 256)
        # 第三层
        # x_SH77 = self.convSH_TostudyS_3_h(x_SH66, edge_index_SH)

        x_SH99 = (x_SH11 + x_SH22 +x_SH66 ) / 3.0
        x_SH99 = self.SH_mlp_1_h(x_SH99)
        x_SH99 = x_SH99.view(1195, -1)
        x_SH99 = self.SH_bn_1_h(x_SH99)
        x_SH99 = self.SH_tanh_1_h(x_SH99)

        # S-S图搭建
        x_ss0 = self.SH_embedding(x_SS.long())
        x_ss0 = x_ss0.squeeze(1)
        x_ss1 = self.convSS(x_ss0.float(), edge_index_SS) # S_S图中 s的嵌入
        x_ss1 = x_ss1.view(390, -1)
        # H-H图搭建
        x_hh0 = self.SH_embedding(x_HH.long())
        x_hh0 = x_hh0.view(-1, 64)
        # x_hh0 = torch.cat((x_hh0.float(), kgOneHot), dim=-1)
        x_hh1 = self.convHH(x_hh0.float(), edge_index_HH)  # H_H图中 h的嵌入
        x_hh1 = x_hh1.view(805, -1)
        # 信息融合
        # sum
        es = x_SH9[:390] + x_ss1  # 1195,1,64  390,1,64
        eh = x_SH99[390:] + x_hh1 # 805*dim
        # cat
        # es = torch.cat((x_SH9[:390], x_ss1), dim=-1)
        # eh = torch.cat((x_SH99[390:], x_hh1), dim=-1)
        # SI 集成多个症状为一个症状表示 batch*390 390*dim => batch*dim
        # print("es_per", es.shape) 390 256
        es = es.view(390,-1)
        # print("es_old", es.shape) 390 256
        es = self.up2pre(es)  # 390 1350
        e_synd = torch.mm(OH, es.T)  # prescription * es    # 64 390
        e_synd = self.down2pre(e_synd)  # 64 256
        # print("e_synd", e_synd.shape)512 256
        # batch*1
        preSum = OH.sum(dim=1).view(-1, 1)
        # print("preSum", preSum.shape) 512 1
        # batch * dim
        e_synd_norm = e_synd / preSum
        e_synd_norm = self.mlp(e_synd_norm)
        e_synd_norm = e_synd_norm.view(-1, 256)
        e_synd_norm = self.SI_bn(e_synd_norm)
        e_synd_norm = self.relu(e_synd_norm)  # batch*dim
        # batch*dim dim*805 => batch*805
        eh = eh.view(805, -1)
        pre = torch.mm(e_synd_norm, eh.t())
        # [512, 805]
        # ----------------------------------
        class_output = self.class_head1(pre)
        regression_output = self.regression_head1(pre)
        class_output = self.class_head2(class_output) 
        regression_output = self.regression_head2(regression_output)
        mask = F.sigmoid(class_output)
        binary_mask = (mask > 0.8).float().detach()
        outputs = {
            'pred_logits': class_output,
            'pred_values': F.relu(regression_output * binary_mask) 
        }
        return outputs








