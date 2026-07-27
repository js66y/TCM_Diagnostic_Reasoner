"""Official RTDL MLP core adapted to the TCM-EHR comparison task.

The MLP class below is retained from:
https://github.com/yandex-research/rtdl-revisiting-models
Paper: Revisiting Deep Learning Models for Tabular Data
https://arxiv.org/abs/2106.11959
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class MLP(nn.Module):
    """The official MLP model from Section 3.1 of the RTDL paper."""

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        dropout: float,
    ) -> None:
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, however: {n_blocks=}")
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _named_sequential(
                    ("linear", nn.Linear(d_block if i else d_in, d_block)),
                    ("activation", nn.ReLU()),
                    ("dropout", nn.Dropout(dropout)),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


class TCMFeatureEncoder(nn.Module):
    """Learned categorical encoding plus normalized continuous age."""

    def __init__(
        self,
        cardinalities: List[int],
        method_dim: int,
        embedding_dim: int,
        max_age: int,
    ) -> None:
        super().__init__()
        if len(cardinalities) != 4:
            raise ValueError(
                "cardinalities must be [gender, preliminary, TCM symptom, syndrome]"
            )
        self.max_age = max_age
        sizes = list(cardinalities) + [method_dim, method_dim, method_dim]
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in sizes
        )
        self.output_dim = 1 + len(sizes) * embedding_dim

    def forward(self, age: Tensor, codes: Tensor) -> Tensor:
        if age.ndim != 2 or age.shape[1] != 1:
            raise ValueError(f"age must have shape [batch, 1], got {tuple(age.shape)}")
        if codes.ndim != 2 or codes.shape[1] != 7:
            raise ValueError(
                f"codes must have shape [batch, 7], got {tuple(codes.shape)}"
            )
        values = [age.float().clamp(0, self.max_age) / float(self.max_age)]
        for index, embedding in enumerate(self.embeddings):
            code = codes[:, index].long()
            if torch.any(code < 0) or torch.any(code >= embedding.num_embeddings):
                raise ValueError(
                    f"codes[:, {index}] contains an ID outside "
                    f"[0, {embedding.num_embeddings - 1}]"
                )
            values.append(embedding(code))
        return torch.cat(values, dim=1)


class MLPTCM(nn.Module):
    """RTDL MLP baseline for 510-label herb and conditional dose prediction."""

    checkpoint_version = "official-rtdl-mlp-tcm-v1"

    def __init__(
        self,
        cardinalities: List[int],
        method_dim: int,
        num_herb: int = 510,
        embedding_dim: int = 32,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        dose_threshold: float = 0.5,
        max_dose: float = 500.0,
        max_age: int = 104,
    ) -> None:
        super().__init__()
        self.num_herb = num_herb
        self.dose_threshold = dose_threshold
        self.max_dose = max_dose
        self.feature_encoder = TCMFeatureEncoder(
            cardinalities, method_dim, embedding_dim, max_age
        )
        # d_out=None exposes the official MLP's final hidden representation.
        self.core = MLP(
            d_in=self.feature_encoder.output_dim,
            d_out=None,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_block, num_herb)
        self.dose_head = nn.Linear(d_block, num_herb)

    def forward(
        self,
        X: Tuple[Tensor, Tensor],
        bank: Optional[Tensor] = None,
        gat: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:
        del bank, gat
        age, codes = X
        representation = self.core(self.feature_encoder(age, codes))
        logits = self.classifier(representation)
        pred_log_values = self.dose_head(representation).clamp(
            min=0.0, max=math.log1p(self.max_dose)
        )
        pred_mask = (torch.sigmoid(logits) >= self.dose_threshold).to(logits.dtype)
        pred_values = torch.expm1(pred_log_values) * pred_mask.detach()
        return {
            "pred_logits": logits,
            "pred_mask": pred_mask,
            "pred_log_values": pred_log_values,
            "pred_values": pred_values.clamp(max=self.max_dose),
            "loss_profile": "mlp",
        }

    def optimizer_parameter_groups(self):
        return self.parameters()


def mlp_tcm_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(prediction, target.float())


def conditional_dose_loss(
    pred_log_values: Tensor,
    target_values: Tensor,
    target_mask: Tensor,
) -> Tensor:
    active = target_mask.bool()
    if not torch.any(active):
        return pred_log_values.sum() * 0.0
    target_log_values = torch.log1p(target_values.float().clamp_min(0.0))
    return F.smooth_l1_loss(
        pred_log_values[active], target_log_values[active]
    )


def _smoke_test() -> None:
    torch.manual_seed(7)
    model = MLPTCM(
        cardinalities=[2, 654, 207, 312],
        method_dim=174,
        embedding_dim=16,
        n_blocks=2,
        d_block=64,
    )
    age = torch.tensor([[34.0], [68.0]])
    codes = torch.tensor(
        [[0, 12, 8, 25, 3, 7, 9], [1, 653, 206, 311, 173, 0, 15]]
    )
    target = torch.zeros(2, 510)
    target[:, :5] = 1
    output = model((age, codes))
    loss = mlp_tcm_loss(output["pred_logits"], target)
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    assert output["pred_logits"].shape == (2, 510)
    assert torch.isfinite(loss)
    print(f"MLP TCM smoke test passed; loss={loss.item():.6f}")


if __name__ == "__main__":
    _smoke_test()
