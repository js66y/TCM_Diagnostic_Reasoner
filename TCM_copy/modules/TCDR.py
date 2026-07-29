import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OneHotEncoding0d(nn.Module):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)
        return torch.cat(
            [
                nn.functional.one_hot(x[..., i], cardinality)
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        ).float()


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

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        W_eff = self.weight * mask
        output = torch.matmul(input, W_eff)
        if self.bias is not None:
            output = output + self.bias
        return output


class MLP(nn.Module):
    def __init__(self, d_in: int, n_blocks: int, d_block: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_in if i == 0 else d_block, d_block),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class StructuredFeatureEncoder(nn.Module):
    """Encode structured TCM fields into a compact learned representation.

    The topology branch still consumes one-hot/multi-hot features.  This encoder
    is an additional data-driven branch and therefore does not change the shape
    of any prior matrix used by ``MaskLinear``.
    """

    def __init__(
        self,
        cardinalities: list[int],
        method_dim: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        max_age: int = 104,
    ) -> None:
        super().__init__()
        if len(cardinalities) != 4:
            raise ValueError(
                "cardinalities must contain gender, preliminary diagnosis, "
                "TCM diagnosis and syndrome pattern sizes"
            )

        base_dim = max(int(emb_dim), 8)
        age_dim = max(base_dim // 4, 8)
        gender_dim = max(base_dim // 8, 4)
        diagnosis_dim = base_dim
        tcm_dim = max(3 * base_dim // 4, 16)
        pattern_dim = max(3 * base_dim // 4, 16)
        method_emb_dim = base_dim

        self.max_age = max(float(max_age), 1.0)
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_dim),
            nn.ReLU(),
        )
        self.gender_embedding = nn.Embedding(cardinalities[0], gender_dim)
        self.pre_embedding = nn.Embedding(cardinalities[1], diagnosis_dim)
        self.tcm_embedding = nn.Embedding(cardinalities[2], tcm_dim)
        self.pattern_embedding = nn.Embedding(cardinalities[3], pattern_dim)
        self.method_embedding = nn.Embedding(method_dim, method_emb_dim)

        encoder_in_dim = (
            age_dim
            + gender_dim
            + diagnosis_dim
            + tcm_dim
            + pattern_dim
            + method_emb_dim
        )
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        age: Tensor,
        categorical: Tensor,
        method_multi_hot: Tensor,
    ) -> Tensor:
        batch_size = categorical.shape[0]
        age_value = age.float().reshape(batch_size, -1)[:, :1]
        age_value = (age_value / self.max_age).clamp(min=0.0, max=1.5)
        age_feature = self.age_encoder(age_value)

        category_ids = categorical[:, :4].long()
        gender_feature = self.gender_embedding(category_ids[:, 0])
        pre_feature = self.pre_embedding(category_ids[:, 1])
        tcm_feature = self.tcm_embedding(category_ids[:, 2])
        pattern_feature = self.pattern_embedding(category_ids[:, 3])

        # ``method_multi_hot`` is already a set representation.  Matrix
        # multiplication followed by normalization averages unique methods and
        # avoids overweighting values repeated only to pad the three ID slots.
        method_weights = method_multi_hot.float()
        method_count = method_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        method_feature = torch.matmul(
            method_weights, self.method_embedding.weight
        ) / method_count

        structured_feature = torch.cat(
            [
                age_feature,
                gender_feature,
                pre_feature,
                tcm_feature,
                pattern_feature,
                method_feature,
            ],
            dim=1,
        )
        return self.encoder(structured_feature)


def _load_matrix(root: str, fname_npy: str, fallback_csv: str | None = None) -> np.ndarray:
    path_npy = os.path.join(root, fname_npy)
    if os.path.exists(path_npy):
        return np.load(path_npy)
    if fallback_csv is not None:
        path_csv = os.path.join(root, fallback_csv)
        if os.path.exists(path_csv):
            return np.loadtxt(path_csv, delimiter=",")
    raise FileNotFoundError(f"Cannot find {fname_npy} or CSV fallback under {root}")

class AttentionBlock(nn.Module):
    """Learnable dynamic attention between multiple tensors"""
    def __init__(self, n_inputs, d_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.att_fc = nn.Linear(d_hidden * n_inputs, n_inputs)

    def forward(self, tensors):
        # tensors: list of tensors [t1, t2, ...], each shape [B, D]
        concat = torch.cat(tensors, dim=-1)             # [B, n_inputs*D]
        att_logits = self.att_fc(concat)                # [B, n_inputs]
        att_weights = F.softmax(att_logits, dim=-1)     # [B, n_inputs]
        att_weights = att_weights.unsqueeze(-1)         # [B, n_inputs, 1]
        stacked = torch.stack(tensors, dim=1)           # [B, n_inputs, D]
        out = torch.sum(att_weights * stacked, dim=1)   # [B, D]
        return out, att_weights


class TCM_Model(nn.Module):
    def __init__(
        self,
        dropout,
        emb_dim,
        cat_cardinalities,   # [2, num_pre_symptoms, num_tcm_symptoms, num_patterns]
        num_herb,            # herb count
        device,
        mask_assoc_dir: str = "binary_matrices_input",   # 用于 pre->tcm->pattern->method 的先验
        graph_assoc_dir: str = "binary_matrices",  # 用于 graph path 的先验
        method_dim: int = 174,
        max_age: int = 104,
        dose_threshold: float = 0.5,
    ):
        super().__init__()
        # self.device = device  # 未在 forward 中直接使用
        self.cat_module = OneHotEncoding0d(cat_cardinalities)

        # 维度
        self.gender_dim = cat_cardinalities[0]
        self.pre_dim = cat_cardinalities[1]
        self.tcm_dim = cat_cardinalities[2]
        self.pattern_dim = cat_cardinalities[3]
        self.method_dim = method_dim
        self.num_herb = num_herb
        self.dose_threshold = dose_threshold

        # OH总维度（1 + sum(cat) + method_dim）= 1350
        self.oh_dim = 1 + sum(cat_cardinalities) + self.method_dim
        self.fusion_dim = 512

        # 原始特征 MLP（用于回归支路）
        self.backbone_original = MLP(d_in=self.oh_dim, n_blocks=1, d_block=self.oh_dim, dropout=0.1)
        # 仅保留正向推理 MLP（用于分类/回归共同的推理特征）
        self.forward_mlp = MLP(d_in=self.oh_dim, n_blocks=1, d_block=self.oh_dim, dropout=dropout)

        # 结构化语义分支：保留拓扑 one-hot 分支的同时，学习低维类别表示。
        self.structured_encoder = StructuredFeatureEncoder(
            cardinalities=cat_cardinalities,
            method_dim=self.method_dim,
            emb_dim=emb_dim,
            hidden_dim=self.fusion_dim,
            dropout=dropout,
            max_age=max_age,
        )

        # 分类 / 回归头（按新语义调整维度）
        self.class_head1 = nn.Linear(self.oh_dim, self.fusion_dim)
        self.class_head2 = nn.Linear(self.fusion_dim, num_herb)
        self.regression_head1 = nn.Linear(self.oh_dim, self.fusion_dim)
        self.regression_head2 = nn.Linear(self.fusion_dim, num_herb)

        # 分类和计量任务分别学习“拓扑特征 / 结构化嵌入”的融合比例。
        self.class_fusion_gate = nn.Linear(2 * self.fusion_dim, self.fusion_dim)
        self.regression_fusion_gate = nn.Linear(2 * self.fusion_dim, self.fusion_dim)
        self._initialize_fusion_gate(self.class_fusion_gate)
        self._initialize_fusion_gate(self.regression_fusion_gate)

        # 图推理链路的 MaskLinear（仅正向）
        self.mask_pre_tcm = MaskLinear(self.pre_dim, self.tcm_dim)
        self.mask_tcm_pattern = MaskLinear(self.tcm_dim, self.pattern_dim)
        self.mask_pattern_method = MaskLinear(self.pattern_dim, self.method_dim)
        self.mask_method_pattern_rev = MaskLinear(self.method_dim, self.pattern_dim)
        self.mask_pattern_tcm_rev = MaskLinear(self.pattern_dim, self.tcm_dim)
        self.mask_tcm_pre_rev = MaskLinear(self.tcm_dim, self.pre_dim)

        # 融合层：拼接后线性映射回原维度（2*length -> length）
        self.fuse_tcm = nn.Linear(2 * self.tcm_dim, self.tcm_dim)
        self.fuse_pattern = nn.Linear(2 * self.pattern_dim, self.pattern_dim)
        self.fuse_method = nn.Linear(2 * self.method_dim, self.method_dim)
        self.fuse_pattern_rev = nn.Linear(2 * self.pattern_dim, self.pattern_dim)
        self.fuse_tcm_rev = nn.Linear(2 * self.tcm_dim, self.tcm_dim)
        self.fuse_pre_rev = nn.Linear(2 * self.pre_dim, self.pre_dim)

        # 下列图增强分支的投影/重构与嵌入未在 forward 使用，先注释以简化
        # self.preliminary_proj = nn.Linear(self.pre_dim, emb_dim)
        # self.tcm_diagnosis_proj = nn.Linear(self.tcm_dim, emb_dim)
        # self.pattern_proj = nn.Linear(self.pattern_dim, emb_dim)
        # self.method_proj = nn.Linear(self.method_dim, emb_dim)
        # self.preliminary_recon = nn.Linear(emb_dim, self.pre_dim)
        # self.tcm_diagnosis_recon = nn.Linear(emb_dim, self.tcm_dim)
        # self.pattern_recon = nn.Linear(emb_dim, self.pattern_dim)
        # self.method_recon = nn.Linear(emb_dim, self.method_dim)
        # self.herb_embedding = nn.Parameter(torch.randn(num_herb, emb_dim))
        # self.herb_bias = nn.Parameter(torch.zeros(num_herb))

        # 图交互/融合权重目前未在 forward 中使用，注释
        # self.alpha_symptom = nn.Parameter(torch.tensor(0.1))
        # self.alpha_pattern = nn.Parameter(torch.tensor(0.1))
        # self.alpha_method = nn.Parameter(torch.tensor(0.1))
        self.w_graph = nn.Parameter(torch.tensor(0.5))
        self.w_infer = nn.Parameter(torch.tensor(0.5))

        # 加载推理链路先验矩阵
        priors = self._load_mask_priors(mask_assoc_dir)
        self.register_buffer("M_pre_tcm", priors["pre_tcm"])
        self.register_buffer("M_tcm_pattern", priors["tcm_pattern"])
        self.register_buffer("M_pattern_method", priors["pattern_method"])

        # 加载图增强分支先验矩阵
        gpriors = self._load_graph_priors(graph_assoc_dir)
        self.register_buffer("A_preliminary_symptoms_herb", gpriors["pre_herb"])
        self.register_buffer("A_TCM_symptoms_herb", gpriors["tcm_herb"])
        self.register_buffer("A_syndrome_pattern_herb", gpriors["pattern_herb"])
        self.register_buffer("A_method_herb", gpriors["method_herb"])
        # self.att = AttentionBlock(3, self.oh_dim)  # 未使用；四路融合用 branch_attn4

        # === 行归一化后的先验矩阵，用于 MaskLinear ===
        self.register_buffer("A_preliminary_symptoms_herb_norm", self._row_normalize(self.A_preliminary_symptoms_herb))
        self.register_buffer("A_TCM_symptoms_herb_norm", self._row_normalize(self.A_TCM_symptoms_herb))
        self.register_buffer("A_syndrome_pattern_herb_norm", self._row_normalize(self.A_syndrome_pattern_herb))
        self.register_buffer("A_method_herb_norm", self._row_normalize(self.A_method_herb))

        # 图增强分支的可学习嵌入与共享解码器
        self.embed_pre = nn.Linear(self.pre_dim, self.pre_dim)
        self.embed_tcm = nn.Linear(self.tcm_dim, self.tcm_dim)
        self.embed_pattern = nn.Linear(self.pattern_dim, self.pattern_dim)
        self.embed_method = nn.Linear(self.method_dim, self.method_dim)
        self.herb_decoder = nn.Linear(self.num_herb, self.num_herb)

        # 四路输入到草药维度的 MaskLinear 头
        self.mask_pre_herb = MaskLinear(self.pre_dim, self.num_herb, bias=False)
        self.mask_tcm_herb = MaskLinear(self.tcm_dim, self.num_herb, bias=False)
        self.mask_pattern_herb = MaskLinear(self.pattern_dim, self.num_herb, bias=False)
        self.mask_method_herb = MaskLinear(self.method_dim, self.num_herb, bias=False)

        # 四路草药证据的注意力融合 + MLP 映射回原维度
        self.branch_attn4 = AttentionBlock(4, self.num_herb)
        self.graph_herb_mlp = MLP(d_in=self.num_herb, n_blocks=1, d_block=self.oh_dim, dropout=0.1)

        # === 输出端 herb-herb 共现残差校正（同维度）===
        self.register_buffer(
            "A_herb_herb",
            torch.tensor(_load_matrix(graph_assoc_dir, "binary_herb_herb.npy", "binary_herb_herb.csv"), dtype=torch.float32)
        )
        self.register_buffer("A_herb_herb_norm", self._row_normalize(self.A_herb_herb))
        self.herb_cooccur_head = MaskLinear(self.num_herb, self.num_herb, bias=False)
        self.lambda_hh = nn.Parameter(torch.tensor(0.1))

    def _load_mask_priors(self, root: str):
        def load(path_npy: str, path_csv: str, expected_shape: tuple[int, int]):
            if os.path.exists(path_npy):
                arr = np.load(path_npy)
            elif os.path.exists(path_csv):
                arr = np.loadtxt(path_csv, delimiter=",")
            else:
                raise FileNotFoundError(f"Prior not found: {path_npy} or {path_csv}")
            if arr.shape != expected_shape:
                raise ValueError(f"Prior shape mismatch: got {arr.shape}, expect {expected_shape}")
            return torch.tensor(arr, dtype=torch.float32)

        pre_tcm_npy = os.path.join(root, "preliminary_symptoms__TCM_symptoms.npy")
        pre_tcm_csv = os.path.join(root, "preliminary_symptoms__TCM_symptoms.csv")
        tcm_pattern_npy = os.path.join(root, "TCM_symptoms__syndrome_pattern.npy")
        tcm_pattern_csv = os.path.join(root, "TCM_symptoms__syndrome_pattern.csv")
        pattern_method_npy = os.path.join(root, "syndrome_pattern__method.npy")
        pattern_method_csv = os.path.join(root, "syndrome_pattern__method.csv")

        return {
            "pre_tcm": load(pre_tcm_npy, pre_tcm_csv, (self.pre_dim, self.tcm_dim)),
            "tcm_pattern": load(tcm_pattern_npy, tcm_pattern_csv, (self.tcm_dim, self.pattern_dim)),
            "pattern_method": load(pattern_method_npy, pattern_method_csv, (self.pattern_dim, self.method_dim)),
        }

    def _load_graph_priors(self, root: str):
        return {
            "pre_herb": torch.tensor(_load_matrix(root, "binary_preliminary_symptoms_herb.npy", "binary_preliminary_symptoms_herb.csv"), dtype=torch.float32),
            "tcm_herb": torch.tensor(_load_matrix(root, "binary_TCM_symptoms_herb.npy", "binary_TCM_symptoms_herb.csv"), dtype=torch.float32),
            "pattern_herb": torch.tensor(_load_matrix(root, "binary_syndrome_pattern_herb.npy", "binary_syndrome_pattern_herb.csv"), dtype=torch.float32),
            "method_herb": torch.tensor(_load_matrix(root, "binary_method_herb.npy", "binary_method_herb.csv"), dtype=torch.float32),
        }

    def _row_normalize(self, A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        row_sum = A.sum(dim=1, keepdim=True)
        inv = (row_sum + eps).reciprocal()
        return A * inv

    @staticmethod
    def _initialize_fusion_gate(gate: nn.Linear) -> None:
        # Start slightly biased toward the established TCDR branch, then let
        # training learn sample-wise contributions from both representations.
        nn.init.zeros_(gate.weight)
        nn.init.ones_(gate.bias)

    @staticmethod
    def _gated_fusion(base_feature: Tensor, semantic_feature: Tensor, gate_layer: nn.Linear) -> Tensor:
        gate = torch.sigmoid(gate_layer(torch.cat([base_feature, semantic_feature], dim=1)))
        return gate * base_feature + (1.0 - gate) * semantic_feature

    def _extract_features_from_OH(self, OH: Tensor):
        s_pre = 1 + self.gender_dim
        s_tcm = s_pre + self.pre_dim
        s_pattern = s_tcm + self.tcm_dim
        s_method = s_pattern + self.pattern_dim

        meta = OH[:, :3]
        pre = OH[:, s_pre:s_tcm]
        tcm = OH[:, s_tcm:s_pattern]
        pattern = OH[:, s_pattern:s_method]
        method = OH[:, s_method:]
        return meta, pre, tcm, pattern, method

    def _graph_feature_interaction(self, OH: Tensor):
        device = OH.device
        # 原四段特征切分
        pre_feat, tcm_feat, pattern_feat, method_feat = self._extract_features_from_OH(OH)[1:]

        # 1) 四路“嵌入”（同维度线性）
        pre_emb = self.embed_pre(pre_feat)             # [B, pre_dim]
        tcm_emb = self.embed_tcm(tcm_feat)             # [B, tcm_dim]
        pattern_emb = self.embed_pattern(pattern_feat) # [B, pattern_dim]
        method_emb = self.embed_method(method_feat)    # [B, method_dim]

        # 2) MaskLinear 到草药维度（受先验掩码约束）
        pre_to_herb = self.mask_pre_herb(pre_emb, self.A_preliminary_symptoms_herb_norm)       # [B, H]
        tcm_to_herb = self.mask_tcm_herb(tcm_emb, self.A_TCM_symptoms_herb_norm)               # [B, H]
        pattern_to_herb = self.mask_pattern_herb(pattern_emb, self.A_syndrome_pattern_herb_norm)  # [B, H]
        method_to_herb = self.mask_method_herb(method_emb, self.A_method_herb_norm)               # [B, H]

        # 3) 共享 decoder 到草药维度（H -> H），提升表达与可学习性
        pre_dec = self.herb_decoder(pre_to_herb)           # [B, H]
        tcm_dec = self.herb_decoder(tcm_to_herb)           # [B, H]
        pattern_dec = self.herb_decoder(pattern_to_herb)   # [B, H]
        method_dec = self.herb_decoder(method_to_herb)     # [B, H]

        # 4) 注意力加权融合四路草药向量
        graph_herb, _ = self.branch_attn4([pre_dec, tcm_dec, pattern_dec, method_dec])  # [B, H]
        return graph_herb

    def forward(self, X, bank, gat):
        # 构造 OH
        x_list = []
        x_list.append(X[0])
        x_list.append(self.cat_module(X[1][:, :4]).float())
        symptom_OH = torch.column_stack([x_.flatten(1, -1) for x_ in x_list])

        dev = symptom_OH.device
        one_hot = torch.zeros(X[1].shape[0], self.method_dim, dtype=torch.float32, device=dev)
        one_hot.scatter_(1, X[1][:, 4:].long(), 1)
        OH = torch.concat((symptom_OH, one_hot), dim=1)  # [B, 1350]

        # 低维结构化嵌入；治法使用已去重的 multi-hot 集合聚合。
        semantic_features = self.structured_encoder(X[0], X[1], one_hot)  # [B, 512]

        # 原始特征（仅用于回归支路）
        original_features = self.backbone_original(OH)  # [B, 1350]

        # 图推理（仅正向）——包含 meta
        meta, pre, tcm, pattern, method = self._extract_features_from_OH(OH)
        tcm_msg = self.mask_pre_tcm(pre, self.M_pre_tcm)
        tcm_fwd = self.fuse_tcm(torch.cat([tcm, tcm_msg], dim=1))
        pattern_msg = self.mask_tcm_pattern(tcm_fwd, self.M_tcm_pattern)
        pattern_fwd = self.fuse_pattern(torch.cat([pattern, pattern_msg], dim=1))
        method_msg = self.mask_pattern_method(pattern_fwd, self.M_pattern_method)
        method_fwd = self.fuse_method(torch.cat([method, method_msg], dim=1))
        forward_OH = torch.cat([meta, pre, tcm_fwd, pattern_fwd, method_fwd], dim=1)  # [B, 1350]
        forward_rep = self.forward_mlp(forward_OH)  # [B, 1350]

        # 图增强（embedding+MaskLinear+decoder+注意力 → 草药维度）
        graph_herb = self._graph_feature_interaction(OH)  # [B, num_herb]

        # 分类：拓扑推理特征与结构化嵌入门控融合，再与图增强结果加和。
        class_topology = self.class_head1(forward_rep)  # [B, 512]
        class_features = self._gated_fusion(
            class_topology, semantic_features, self.class_fusion_gate
        )
        class_infer = self.class_head2(class_features)  # [B, num_herb]
        class_output = self.w_infer * class_infer + self.w_graph * graph_herb       # [B, num_herb]

        # herb-herb 共现残差校正
        class_output = class_output + torch.clamp(self.lambda_hh, min=0.0) * \
            self.herb_cooccur_head(class_output, self.A_herb_herb_norm)

        # 分类掩码引导计量输出。训练时使用停止梯度的软掩码，
        # 避免随机初始化阶段所有概率都未过硬阈值，导致回归分支零梯度。
        mask_cls = torch.sigmoid(class_output)
        if self.training:
            dose_mask = mask_cls.detach()
        else:
            dose_mask = (mask_cls > self.dose_threshold).float()

        # 回归：原始结构特征与结构化嵌入独立门控融合。
        regression_original = self.regression_head1(original_features)  # [B, 512]
        regression_features = self._gated_fusion(
            regression_original, semantic_features, self.regression_fusion_gate
        )
        regression_output = self.regression_head2(regression_features)  # [B, num_herb]

        outputs = {
            "pred_logits": class_output,
            "pred_values": F.relu(regression_output * dose_mask),
        }
        return outputs
