
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
        )

class LeapStyleAggregator(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.emb_dim = emb_dim
        # 分组编码：g0 = base(1) + cat(864) = 865, g1 = method(174)
        self.enc_g0 = nn.Linear(865, emb_dim, bias=False)
        self.enc_g1 = nn.Linear(485, emb_dim, bias=False)
        # 加性注意力：scores = v^T tanh(W h_i + q)
        self.att_W = nn.Linear(emb_dim, emb_dim, bias=True)
        self.att_v = nn.Linear(emb_dim, 1, bias=False)
        self.query = nn.Parameter(torch.randn(emb_dim))

    def forward(self, OH: Tensor) -> Tensor:
        # OH: (B, 1039) = concat([base+cat(865), method(174)])
        g0 = OH[:, :865]
        g1 = OH[:, 865:]
        e0 = self.enc_g0(g0)     # (B, E)
        e1 = self.enc_g1(g1)     # (B, E)
        steps = torch.stack([e0, e1], dim=1)  # (B, 2, E)

        q = self.query.view(1, 1, -1).expand(steps.size(0), 1, -1)  # (B, 1, E)
        scores = self.att_v(torch.tanh(self.att_W(steps) + q))      # (B, 2, 1)
        attn = torch.softmax(scores, dim=1)                         # (B, 2, 1)
        context = (attn * steps).sum(dim=1)                         # (B, E)
        return context

class TCMLeapModel(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],  # 例如 [1, 2, 654, 207] -> 总和为 864
        num_herb: int,                 # 例如 510
        emb_dim: int = 256,
        mask_threshold: float = 0.8,
    ):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.cat_module = OneHotEncoding0d(cat_cardinalities)
        self.leap_agg = LeapStyleAggregator(emb_dim=emb_dim)
        # 输出头：与 PresRecST_copy.py 的 L140-150 等价，只把输入改为 emb_dim
        self.class_head1 = nn.Linear(emb_dim, 512)
        self.regression_head1 = nn.Linear(emb_dim, 512)
        self.class_head2 = nn.Linear(512, num_herb)
        self.regression_head2 = nn.Linear(512, num_herb)

    def forward(self, X, bank=None, gat=None):
        # 与 PresRecST_copy.py#L122-128 相同的输入构造
        x = []
        x.append(X[0])  # (B, 1)
        x.append(self.cat_module(X[1][:, :4].long()).float())  # (B, 864)
        symptom_OH = torch.column_stack([x_.flatten(1, -1) for x_ in x])  # (B, 865)

        device = X[1].device
        B = X[1].shape[0]
        idx = X[1][:, 4:].long()                           # (B, 174) 多个 method 索引（可能含填充）
        valid = (idx >= 0) & (idx < 174)                   # 0 基索引过滤；如为 1 基需改为 (idx > 0) & (idx <= 174) 且 cols-1
        rows = torch.arange(B, device=device).unsqueeze(1).expand_as(idx)[valid]
        cols = idx[valid]
        method_oh = torch.zeros(B, 174, dtype=torch.float32, device=device)
        method_oh[rows, cols] = 1.0

        OH = torch.concat((symptom_OH, method_oh), dim=1)  # (B, 1039)

        # 用 LEAP 聚合器替代 MLP 的特征抽取
        query_output = self.leap_agg(OH)

        # 输出与 PresRecST_copy.py#L140-150 等价
        class_output = self.class_head1(query_output)
        regression_output = self.regression_head1(query_output)
        class_output = self.class_head2(class_output)
        regression_output = self.regression_head2(regression_output)
        mask = torch.sigmoid(class_output)
        binary_mask = (mask > self.mask_threshold).float().detach()
        outputs = {
            "pred_logits": class_output,
            "pred_values": F.relu(regression_output * binary_mask),
        }
        return outputs