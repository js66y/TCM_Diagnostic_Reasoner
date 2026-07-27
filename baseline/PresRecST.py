import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
from torch import Tensor
import torch.nn.functional as F
import math
# from torch_geometric.nn import AttentiveFP

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

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


# class TCM_Model(torch.nn.Module):
#     def __init__(self, dropout, emb_dim, cat_cardinalities, num_herb, device):
#         super(TCM_Model, self).__init__()
#         self.batch_size = 64
#         self.embedding_dim = emb_dim
#         self.sym_len, self.syn_len, self.treat_len, self.herb_len = 864, 312, 174, 510
#         self.sym_embedding = torch.nn.Embedding(self.sym_len, self.embedding_dim)
#         self.syn_embedding = torch.nn.Embedding(self.syn_len, self.embedding_dim)
#         self.treat_embedding = torch.nn.Embedding(self.treat_len, self.embedding_dim)
#         self.herb_embedding = torch.nn.Embedding(self.herb_len, self.embedding_dim)
#         # print(self.sym_embedding.weight.shape, self.syn_embedding.weight.shape, self.treat_embedding.weight.shape, self.herb_embedding.weight.shape)
#         # exit(0)
#         # [974, 64],[50, 64],[61, 64],[379, 64]

#         self.mlp_sym = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.mlp_syn_1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.mlp_syn_2 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim)
#         self.mlp_treat_1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.mlp_treat_2 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim)
#         self.mlp_treat_3 = torch.nn.Linear(self.embedding_dim*3, self.embedding_dim)
#         self.relu = torch.nn.ReLU()
#         self.batch_norm = torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim)

#         self.bn_layer1 = torch.nn.Sequential(
#             torch.nn.Linear(self.embedding_dim, self.embedding_dim),
#             torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.embedding_dim, self.embedding_dim)
#         )

#         self.bn_layer2 = torch.nn.Sequential(
#             torch.nn.Linear(2*self.embedding_dim, self.embedding_dim),
#             torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.embedding_dim, self.embedding_dim)
#         )

#         self.bn_layer3 = torch.nn.Sequential(
#             torch.nn.Linear(3*self.embedding_dim, self.embedding_dim),
#             torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.embedding_dim, self.embedding_dim)
#         )

#         self.cat_module = OneHotEncoding0d(cat_cardinalities)
        
#         # self.bank_encoder = GNNGraph(num_layer=2, emb_dim=emb_dim, graph_pooling='mean', drop_ratio=0.7, gnn_type='gin', virtual_node=False)
#         # self.bank_encoder = AttentiveFP(in_channels=9,hidden_channels=200, out_channels=8867, edge_dim=3, num_layers=3, num_timesteps=2)



#         self.class_head1 = nn.Linear(510, 512)
#         self.regression_head1 = nn.Linear(510, 512)
#         self.class_head2 = nn.Linear(512, num_herb)
#         self.regression_head2 = nn.Linear(512, num_herb)
#         self.down1 = MaskLinear(864, 312)
#         self.down2 = MaskLinear(312, 174)
#         self.relu1 = nn.Sequential(
#             nn.Linear(864, 864),
#             nn.ReLU()
#         )
#         self.relu2 = nn.Sequential(
#             nn.Linear(312, 312),
#             nn.ReLU()
#         )

#     def forward(self, X, bank, gat):
#         # print(gat[0].shape, gat[1].shape)
#         gat0 = torch.FloatTensor(gat[0]).to("cuda:0")
#         gat1 = torch.FloatTensor(gat[1]).to("cuda:0")
#         # exit(0)
#         # X 1+7[2+654+207+312+174+174+174]
#         x = []
#         x.append(X[0])  # 64,1-64,7
#         x.append(self.cat_module(X[1]).float())
#         symptom_OH = torch.column_stack([x_.flatten(1, -1) for x_ in x])  # 64,864
        
#         # 1. symptom embedding
#         get_sym = torch.mm(symptom_OH[:, :864], self.sym_embedding.weight)  # 64, 64
#         sym_agg = self.mlp_sym(get_sym) # 64, 64
#         sym_agg = self.bn_layer1(sym_agg) #64, 64

#         dd = self.relu1(symptom_OH[:, :864])
#         down1 = self.down1(dd, gat0) # 64, 312
        

#         # 2. syndrome embedding
#         judge_syndrome = torch.mm(sym_agg, down1) # 64, 312
#         get_syn = torch.mm(judge_syndrome, down1.T)

#         cat_sym_syn = torch.cat((sym_agg, get_syn), dim=-1) # 64, 64
#         syn_agg = self.bn_layer2(cat_sym_syn) # 64, 64
#         # print(syn_agg.shape)
#         # print(symptom_OH[:, 864:864 + 312].shape, gat1.shape)

#         dd = self.relu2(symptom_OH[:, 864:864 + 312])
#         down2 = self.down2(dd, gat1) # 64, 174

#         # 3. treat embedding
#         judge_treatment = torch.mm(syn_agg, down2) # 64, 174
#         get_treat = torch.mm(judge_treatment, down2.T) # 64, 64

#         cat_syn_treat = torch.cat((get_syn, get_treat), dim=-1) # 64,128
#         treat_agg = self.mlp_treat_2(cat_syn_treat) # 64,64

#         cat_syn_treat = torch.cat((sym_agg, syn_agg, treat_agg), dim=-1) # 64, 192
#         treat_agg = self.bn_layer3(cat_syn_treat) # 64,64
#         # print(treat_agg.shape)
#         # exit(0)

#         # 4. judge herb
#         judge_herb = torch.mm(treat_agg, self.herb_embedding.weight.T) # 64,510
#         # ----------------------------------
#         # bank_emb = self.bank_encoder(bank[0])
#         # bank_emb = torch.mm(bank[1].float().to("cuda:0"), bank_emb)  # [num_herb, emb_dim]

#         # 简化查询机制
#         # 方案1: 直接使用judge_herb (最简单，效果最好)
#         query_output = judge_herb  # [64, 510]
        
#         # 方案2: 残差连接 (如果需要融入bank_emb信息)
#         # bank_enhanced = torch.mm(torch.mm(judge_herb, bank_emb), bank_emb.T)  # [64, 510]
#         # query_output = judge_herb + 0.1 * bank_enhanced  # 残差连接，权重可调
        
#         # 方案3: 门控机制 (更复杂但可能更有效)
#         # gate = torch.sigmoid(torch.mm(judge_herb, bank_emb).mean(dim=-1, keepdim=True))  # [64, 1]
#         # bank_info = torch.mm(torch.mm(judge_herb, bank_emb), bank_emb.T)  # [64, 510]
#         # query_output = gate * bank_info + (1 - gate) * judge_herb  # [64, 510]
#         # print(bank[0].x.shape[1], bank[0].edge_attr.shape[1])
#         # exit(0)
        
#         # ----------------------------------
#         class_output = self.class_head1(query_output)
#         regression_output = self.regression_head1(query_output)
#         class_output = self.class_head2(class_output) 
#         regression_output = self.regression_head2(regression_output)
#         mask = F.sigmoid(class_output)
#         binary_mask = (mask > 0.8).float().detach()
#         outputs = {
#             'pred_logits': class_output,
#             'pred_values': F.relu(regression_output * binary_mask) 
#         }
#         return outputs
    




class TCM_Model(torch.nn.Module):
    def __init__(self, dropout, emb_dim, cat_cardinalities, num_herb, device):
        super(TCM_Model, self).__init__()
        self.device = device
        self.batch_size = 64
        self.embedding_dim = 64
        self.sym_len, self.syn_len, self.treat_len, self.herb_len = 864, 312, 174, 510
        self.sym_embedding = torch.nn.Embedding(self.sym_len, self.embedding_dim)
        self.syn_embedding = torch.nn.Embedding(self.syn_len, self.embedding_dim)
        self.treat_embedding = torch.nn.Embedding(self.treat_len, self.embedding_dim)
        self.herb_embedding = torch.nn.Embedding(self.herb_len, self.embedding_dim)
        self.mlp_sym = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mlp_syn_1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mlp_syn_2 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.mlp_treat_1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mlp_treat_2 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.mlp_treat_3 = torch.nn.Linear(self.embedding_dim*3, self.embedding_dim)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim)

        self.bn_layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.bn_layer2 = torch.nn.Sequential(
            torch.nn.Linear(2*self.embedding_dim, self.embedding_dim),
            torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.bn_layer3 = torch.nn.Sequential(
            torch.nn.Linear(3*self.embedding_dim, self.embedding_dim),
            torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.cat_module = OneHotEncoding0d(cat_cardinalities)
        self.class_head1 = nn.Linear(510, 512)
        self.regression_head1 = nn.Linear(510, 512)
        self.class_head2 = nn.Linear(512, num_herb)
        self.regression_head2 = nn.Linear(512, num_herb)

    # def forward(self, symptom_OH):
    
    def forward(self, X, bank, gat):
        x = []
        x.append(X[0])
        x.append(self.cat_module(X[1][:,:4]).float())
        symptom_OH = torch.column_stack([x_.flatten(1, -1) for x_ in x])
        one_hot = torch.zeros(X[1].shape[0], 174, dtype=torch.float32, device=self.device)
        one_hot.scatter_(1, X[1][:, 4:].long(), 1)                #             1   2   3
        symptom_OH = torch.concat((symptom_OH, one_hot), dim=1) # 64, 1350 = 1+2+654+207+312+174


        # 1. symptom embedding
        get_sym = torch.mm(symptom_OH[:, :864], self.sym_embedding.weight)
        sym_agg = self.mlp_sym(get_sym)
        sym_agg = self.bn_layer1(sym_agg)

        # 2. syndrome embedding
        judge_syndrome = torch.mm(sym_agg, self.syn_embedding.weight.T)
        get_syn = torch.mm(judge_syndrome, self.syn_embedding.weight)

        cat_sym_syn = torch.cat((sym_agg, get_syn), dim=-1)
        syn_agg = self.bn_layer2(cat_sym_syn)

        # 3. treat embedding
        judge_treatment = torch.mm(syn_agg, self.treat_embedding.weight.T)
        get_treat = torch.mm(judge_treatment, self.treat_embedding.weight)

        cat_syn_treat = torch.cat((get_syn, get_treat), dim=-1)
        treat_agg = self.mlp_treat_2(cat_syn_treat)

        cat_syn_treat = torch.cat((sym_agg, syn_agg, treat_agg), dim=-1)  # ADD C  两个 20 => 64*256
        treat_agg = self.bn_layer3(cat_syn_treat)

        # 4. judge herb
        judge_herb = torch.mm(treat_agg, self.herb_embedding.weight.T)  # 20*64 * 64*410 => 20*410 每个人对每个中药判断

        # return judge_syndrome, judge_treatment, judge_herb
        query_output = judge_herb
        # ----------------------------------
        class_output = self.class_head1(query_output)
        regression_output = self.regression_head1(query_output)
        class_output = self.class_head2(class_output) 
        regression_output = self.regression_head2(regression_output)
        mask = F.sigmoid(class_output)
        binary_mask = (mask > 0.8).float().detach()
        outputs = {
            'pred_logits': class_output,
            'pred_values': F.relu(regression_output * binary_mask) 
        }
        return outputs
