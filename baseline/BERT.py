"""BERT encoder adapted to the TCM-EHR herb recommendation task.

Input:
    age:   Float/LongTensor [batch, 1]
    codes: LongTensor [batch, 7]
           gender, preliminary symptom, TCM symptom, syndrome,
           treatment method 1/2/3

Output keys are compatible with the existing TCM project:
    pred_logits, pred_mask, pred_log_values, pred_values, loss_profile

The encoder keeps the defining BERT network components (bidirectional
self-attention, post-LayerNorm residual blocks, GELU FFN and CLS pooler).
The original word/position/segment embedding front-end is replaced only by a
TCM structured-field tokenizer. This model is trained from scratch on TCM-EHR;
English BERT checkpoints are intentionally not loaded because their word
vocabulary does not correspond to TCM categorical codes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BertEmbeddingsTCM(nn.Module):
    """TCM value, position and field embeddings in the BERT embedding layout."""

    def __init__(
        self,
        cardinalities: List[int],
        method_dim: int,
        max_age: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(cardinalities) != 4:
            raise ValueError(
                "cardinalities must be [gender, preliminary, TCM symptom, syndrome]"
            )
        cat_sizes = list(cardinalities) + [method_dim, method_dim, method_dim]
        self.max_age = max_age
        self.age_embeddings = nn.Embedding(max_age + 1, hidden_size)
        self.value_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_size) for size in cat_sizes
        )
        # 0=CLS, 1=age, 2..8=the seven categorical fields.
        self.position_embeddings = nn.Embedding(9, hidden_size)
        self.field_embeddings = nn.Embedding(9, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, age: Tensor, codes: Tensor) -> Tensor:
        if age.ndim != 2 or age.shape[1] != 1:
            raise ValueError(f"age must have shape [batch, 1], got {tuple(age.shape)}")
        if codes.ndim != 2 or codes.shape[1] != 7:
            raise ValueError(
                f"codes must have shape [batch, 7], got {tuple(codes.shape)}"
            )
        batch_size = age.shape[0]
        age_ids = age.long().squeeze(1).clamp_(0, self.max_age)
        tokens = [
            self.cls_token.expand(batch_size, -1, -1),
            self.age_embeddings(age_ids).unsqueeze(1),
        ]
        for index, embedding in enumerate(self.value_embeddings):
            value = codes[:, index].long()
            if torch.any(value < 0) or torch.any(value >= embedding.num_embeddings):
                raise ValueError(
                    f"codes[:, {index}] contains an ID outside "
                    f"[0, {embedding.num_embeddings - 1}]"
                )
            tokens.append(embedding(value).unsqueeze(1))
        hidden = torch.cat(tokens, dim=1)
        positions = torch.arange(9, device=hidden.device).unsqueeze(0)
        hidden = (
            hidden
            + self.position_embeddings(positions)
            + self.field_embeddings(positions)
        )
        return self.dropout(self.layer_norm(hidden))


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return (
            x.view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, hidden: Tensor) -> Tensor:
        query = self._split_heads(self.query(hidden))
        key = self._split_heads(self.key(hidden))
        value = self._split_heads(self.value(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        probabilities = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        return context.view(hidden.shape[0], hidden.shape[1], -1)


class BertLayer(nn.Module):
    """Original BERT post-LayerNorm attention and feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = BertSelfAttention(
            hidden_size, num_heads, attention_dropout
        )
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden: Tensor) -> Tensor:
        attended = self.attention_output(self.attention(hidden))
        hidden = self.attention_norm(hidden + self.attention_dropout(attended))
        feed_forward = self.output(F.gelu(self.intermediate(hidden)))
        return self.output_norm(hidden + self.output_dropout(feed_forward))


class BERTTCM(nn.Module):
    """BERT-style encoder for 510-label herb and conditional dose prediction."""

    checkpoint_version = "bert-tcm-v1"

    def __init__(
        self,
        cardinalities: List[int],
        method_dim: int,
        num_herb: int = 510,
        hidden_size: int = 192,
        num_heads: int = 8,
        num_layers: int = 4,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        dose_threshold: float = 0.5,
        max_dose: float = 500.0,
        max_age: int = 104,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size
        self.num_herb = num_herb
        self.dose_threshold = dose_threshold
        self.max_dose = max_dose
        self.embeddings = BertEmbeddingsTCM(
            cardinalities, method_dim, max_age, hidden_size, dropout
        )
        self.encoder = nn.ModuleList(
            BertLayer(
                hidden_size,
                num_heads,
                intermediate_size,
                dropout,
                attention_dropout,
            )
            for _ in range(num_layers)
        )
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_herb)
        self.dose_head = nn.Linear(hidden_size, num_herb)
        self.apply(self._init_bert_weights)

    @staticmethod
    def _init_bert_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        X: Tuple[Tensor, Tensor],
        bank: Optional[Tensor] = None,
        gat: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:
        del bank, gat
        age, codes = X
        hidden = self.embeddings(age, codes)
        for layer in self.encoder:
            hidden = layer(hidden)
        pooled = torch.tanh(self.pooler(hidden[:, 0]))
        logits = self.classifier(pooled)
        pred_log_values = self.dose_head(pooled)
        pred_log_values = pred_log_values.clamp(
            min=0.0, max=math.log1p(self.max_dose)
        )
        pred_mask = (torch.sigmoid(logits) >= self.dose_threshold).to(logits.dtype)
        pred_values = torch.expm1(pred_log_values) * pred_mask.detach()
        return {
            "pred_logits": logits,
            "pred_mask": pred_mask,
            "pred_log_values": pred_log_values,
            "pred_values": pred_values.clamp(max=self.max_dose),
            "loss_profile": "bert",
        }

    def optimizer_parameter_groups(self):
        no_decay = ("bias", "layer_norm.weight", "_norm.weight")
        decay, no_decay_params = [], []
        for name, parameter in self.named_parameters():
            (no_decay_params if any(x in name for x in no_decay) else decay).append(
                parameter
            )
        return [
            {"params": decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]


def bert_tcm_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """The TCM experiment's 510-label herb recommendation objective."""
    return F.binary_cross_entropy_with_logits(prediction, target.float())


def conditional_dose_loss(
    pred_log_values: Tensor,
    target_values: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Smooth-L1 dose loss evaluated only on herbs present in the prescription."""
    active = target_mask.bool()
    if not torch.any(active):
        return pred_log_values.sum() * 0.0
    target_log_values = torch.log1p(target_values.float().clamp_min(0.0))
    return F.smooth_l1_loss(
        pred_log_values[active], target_log_values[active]
    )


def _smoke_test() -> None:
    torch.manual_seed(7)
    model = BERTTCM(
        cardinalities=[2, 654, 207, 312],
        method_dim=174,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
    )
    age = torch.tensor([[34.0], [68.0]])
    codes = torch.tensor(
        [[0, 12, 8, 25, 3, 7, 9], [1, 653, 206, 311, 173, 0, 15]]
    )
    target = torch.zeros(2, 510)
    target[:, :5] = 1
    output = model((age, codes))
    loss = bert_tcm_loss(output["pred_logits"], target)
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    assert output["pred_logits"].shape == (2, 510)
    assert torch.isfinite(loss)
    print(f"BERT TCM smoke test passed; loss={loss.item():.6f}")


if __name__ == "__main__":
    _smoke_test()
