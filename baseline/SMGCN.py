"""Paper-faithful SMGCN reconstruction adapted to TCM-EHR.

Paper:
    Syndrome-aware Herb Recommendation with Multi-Graph Convolution Network
    Yuanyuan Jin et al., ICDE 2020
    https://arxiv.org/abs/2002.08575

No author-maintained public repository could be located as of 2026-07-26.
Consequently, this is explicitly a clean PyTorch reconstruction of the paper's
core architecture, not redistributed official source code.

TCM adaptation:
    * feature-herb bipartite graph: preliminary symptom / TCM symptom /
      syndrome pattern / treatment method to 510 herbs;
    * feature synergy graph: two-hop feature relation induced by shared herbs;
    * herb synergy graph: binary_herb_herb.npy;
    * syndrome induction MLP over age and seven structured fields;
    * syndrome-herb interaction produces 510 recommendation logits;
    * a conditional dose head adds the project's dosage task.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _normalize_bipartite(adjacency: Tensor) -> Tensor:
    adjacency = adjacency.float()
    row_degree = adjacency.sum(dim=1).clamp_min(1.0).rsqrt()
    column_degree = adjacency.sum(dim=0).clamp_min(1.0).rsqrt()
    return row_degree[:, None] * adjacency * column_degree[None, :]


def _normalize_square(adjacency: Tensor) -> Tensor:
    adjacency = adjacency.float()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("A homogeneous graph adjacency must be square")
    adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))
    adjacency = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
    degree = adjacency.sum(dim=1).clamp_min(1.0).rsqrt()
    return degree[:, None] * adjacency * degree[None, :]


def _load_binary_matrix(path: Path, expected_shape: Tuple[int, int]) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SMGCN graph: {path}")
    matrix = np.load(path)
    if matrix.shape == expected_shape[::-1]:
        matrix = matrix.T
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{path.name} has shape {matrix.shape}, expected {expected_shape}"
        )
    return torch.from_numpy((matrix > 0).astype(np.float32, copy=False))


class GraphConvolution(nn.Module):
    """A shared linear graph message transform used by the SMGCN branches."""

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, self_embedding: Tensor, message: Tensor) -> Tensor:
        return torch.tanh(self.linear(self_embedding + self.dropout(message)))


class SMGCNTCM(nn.Module):
    """SMGCN recommendation core plus TCM-EHR input and dose adapters."""

    checkpoint_version = "paper-reconstruction-smgcn-tcm-v1"

    def __init__(
        self,
        cardinalities: List[int],
        method_dim: int,
        num_herb: int = 510,
        embedding_dim: int = 64,
        graph_dir: Optional[Union[str, Path]] = None,
        feature_herb_graph: Optional[Tensor] = None,
        herb_herb_graph: Optional[Tensor] = None,
        dropout: float = 0.0,
        dose_threshold: float = 0.5,
        max_dose: float = 500.0,
        max_age: int = 104,
    ) -> None:
        super().__init__()
        if len(cardinalities) != 4:
            raise ValueError(
                "cardinalities must be [gender, preliminary, TCM symptom, syndrome]"
            )
        self.cardinalities = list(cardinalities)
        self.method_dim = method_dim
        self.num_herb = num_herb
        self.max_age = max_age
        self.dose_threshold = dose_threshold
        self.max_dose = max_dose

        # Groups: age, gender, preliminary symptom, TCM symptom, syndrome, method.
        group_sizes = [
            max_age + 1,
            cardinalities[0],
            cardinalities[1],
            cardinalities[2],
            cardinalities[3],
            method_dim,
        ]
        offsets = [0]
        for size in group_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("feature_offsets", torch.tensor(offsets, dtype=torch.long))
        self.num_feature_nodes = sum(group_sizes)

        if graph_dir is not None and feature_herb_graph is not None:
            raise ValueError("Use graph_dir or feature_herb_graph, not both")
        if graph_dir is not None:
            feature_herb_graph, loaded_herb_graph = self._graphs_from_directory(
                Path(graph_dir)
            )
            if herb_herb_graph is None:
                herb_herb_graph = loaded_herb_graph
        if feature_herb_graph is None:
            # Allows model construction before graphs are prepared. Graph residual
            # paths remain trainable; real comparisons should pass graph_dir.
            feature_herb_graph = torch.zeros(self.num_feature_nodes, num_herb)
        if herb_herb_graph is None:
            herb_herb_graph = torch.zeros(num_herb, num_herb)
        if tuple(feature_herb_graph.shape) != (self.num_feature_nodes, num_herb):
            raise ValueError(
                "feature_herb_graph must have shape "
                f"[{self.num_feature_nodes}, {num_herb}]"
            )
        if tuple(herb_herb_graph.shape) != (num_herb, num_herb):
            raise ValueError(
                f"herb_herb_graph must have shape [{num_herb}, {num_herb}]"
            )

        self.register_buffer(
            "feature_herb_adjacency",
            _normalize_bipartite(feature_herb_graph),
        )
        self.register_buffer(
            "herb_herb_adjacency",
            _normalize_square(herb_herb_graph),
        )
        self.feature_embeddings = nn.Parameter(
            torch.empty(self.num_feature_nodes, embedding_dim)
        )
        self.herb_embeddings = nn.Parameter(torch.empty(num_herb, embedding_dim))
        nn.init.xavier_uniform_(self.feature_embeddings)
        nn.init.xavier_uniform_(self.herb_embeddings)

        # Bipartite GCN branch (feature <-> herb).
        self.feature_bipartite_1 = GraphConvolution(embedding_dim, dropout)
        self.herb_bipartite_1 = GraphConvolution(embedding_dim, dropout)
        self.feature_bipartite_2 = GraphConvolution(embedding_dim, dropout)
        self.herb_bipartite_2 = GraphConvolution(embedding_dim, dropout)
        # Homogeneous synergy branches.
        self.feature_synergy = GraphConvolution(embedding_dim, dropout)
        self.herb_synergy = GraphConvolution(embedding_dim, dropout)
        self.feature_fusion = nn.Linear(2 * embedding_dim, embedding_dim)
        self.herb_fusion = nn.Linear(2 * embedding_dim, embedding_dim)

        # Eight selected fields induce a latent syndrome representation.
        self.syndrome_induction = nn.Sequential(
            nn.Linear(8 * embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Tanh(),
        )
        self.herb_bias = nn.Parameter(torch.zeros(num_herb))
        self.dose_head = nn.Linear(embedding_dim, num_herb)

    def _graphs_from_directory(self, graph_dir: Path) -> Tuple[Tensor, Tensor]:
        feature_herb = torch.zeros(self.num_feature_nodes, self.num_herb)
        # Age and gender have no corresponding aggregate graph file.
        graph_specs = [
            (
                2,
                "binary_preliminary_symptoms_herb.npy",
                self.cardinalities[1],
            ),
            (3, "binary_TCM_symptoms_herb.npy", self.cardinalities[2]),
            (4, "binary_syndrome_pattern_herb.npy", self.cardinalities[3]),
            (5, "binary_method_herb.npy", self.method_dim),
        ]
        for group_index, filename, row_count in graph_specs:
            matrix = _load_binary_matrix(
                graph_dir / filename, (row_count, self.num_herb)
            )
            start = int(self.feature_offsets[group_index].item())
            feature_herb[start : start + row_count] = matrix
        herb_herb = _load_binary_matrix(
            graph_dir / "binary_herb_herb.npy",
            (self.num_herb, self.num_herb),
        )
        return feature_herb, herb_herb

    def _encode_graphs(self) -> Tuple[Tensor, Tensor]:
        feature = self.feature_embeddings
        herb = self.herb_embeddings
        adjacency = self.feature_herb_adjacency

        feature_bip_1 = self.feature_bipartite_1(
            feature, torch.matmul(adjacency, herb)
        )
        herb_bip_1 = self.herb_bipartite_1(
            herb, torch.matmul(adjacency.transpose(0, 1), feature)
        )
        feature_bip_2 = self.feature_bipartite_2(
            feature_bip_1, torch.matmul(adjacency, herb_bip_1)
        )
        herb_bip_2 = self.herb_bipartite_2(
            herb_bip_1, torch.matmul(adjacency.transpose(0, 1), feature_bip_1)
        )

        # The TCM data provides feature-herb priors but no separate symptom-
        # symptom file. Shared-herb two-hop propagation forms its synergy graph.
        feature_two_hop = torch.matmul(
            adjacency, torch.matmul(adjacency.transpose(0, 1), feature)
        )
        feature_synergy = self.feature_synergy(feature, feature_two_hop)
        herb_synergy = self.herb_synergy(
            herb, torch.matmul(self.herb_herb_adjacency, herb)
        )
        feature_final = torch.tanh(
            self.feature_fusion(torch.cat([feature_bip_2, feature_synergy], dim=-1))
        )
        herb_final = torch.tanh(
            self.herb_fusion(torch.cat([herb_bip_2, herb_synergy], dim=-1))
        )
        return feature_final, herb_final

    def _selected_feature_ids(self, age: Tensor, codes: Tensor) -> Tensor:
        if age.ndim != 2 or age.shape[1] != 1:
            raise ValueError(f"age must have shape [batch, 1], got {tuple(age.shape)}")
        if codes.ndim != 2 or codes.shape[1] != 7:
            raise ValueError(
                f"codes must have shape [batch, 7], got {tuple(codes.shape)}"
            )
        age_id = age.long().squeeze(1).clamp(0, self.max_age)
        fields = [
            age_id + self.feature_offsets[0],
            codes[:, 0].long() + self.feature_offsets[1],
            codes[:, 1].long() + self.feature_offsets[2],
            codes[:, 2].long() + self.feature_offsets[3],
            codes[:, 3].long() + self.feature_offsets[4],
            codes[:, 4].long() + self.feature_offsets[5],
            codes[:, 5].long() + self.feature_offsets[5],
            codes[:, 6].long() + self.feature_offsets[5],
        ]
        ids = torch.stack(fields, dim=1)
        if torch.any(ids < 0) or torch.any(ids >= self.num_feature_nodes):
            raise ValueError("Input contains a categorical ID outside its vocabulary")
        return ids

    def forward(
        self,
        X: Tuple[Tensor, Tensor],
        bank: Optional[Tensor] = None,
        gat: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:
        del bank, gat
        age, codes = X
        feature_embeddings, herb_embeddings = self._encode_graphs()
        selected = feature_embeddings[self._selected_feature_ids(age, codes)]
        syndrome = self.syndrome_induction(selected.flatten(1))
        logits = torch.matmul(syndrome, herb_embeddings.transpose(0, 1))
        logits = logits + self.herb_bias
        pred_log_values = self.dose_head(syndrome).clamp(
            min=0.0, max=math.log1p(self.max_dose)
        )
        pred_mask = (torch.sigmoid(logits) >= self.dose_threshold).to(logits.dtype)
        pred_values = torch.expm1(pred_log_values) * pred_mask.detach()
        return {
            "pred_logits": logits,
            "pred_mask": pred_mask,
            "pred_log_values": pred_log_values,
            "pred_values": pred_values.clamp(max=self.max_dose),
            "loss_profile": "smgcn",
        }

    def optimizer_parameter_groups(self):
        return self.parameters()


def smgcn_tcm_loss(prediction: Tensor, target: Tensor) -> Tensor:
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


def _smoke_test(graph_dir: Optional[str] = None) -> None:
    torch.manual_seed(7)
    model = SMGCNTCM(
        cardinalities=[2, 654, 207, 312],
        method_dim=174,
        embedding_dim=32,
        graph_dir=graph_dir,
    )
    age = torch.tensor([[34.0], [68.0]])
    codes = torch.tensor(
        [[0, 12, 8, 25, 3, 7, 9], [1, 653, 206, 311, 173, 0, 15]]
    )
    target = torch.zeros(2, 510)
    target[:, :5] = 1
    output = model((age, codes))
    loss = smgcn_tcm_loss(output["pred_logits"], target)
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=2e-4).step()
    assert output["pred_logits"].shape == (2, 510)
    assert torch.isfinite(loss)
    source = graph_dir or "zero fallback graphs"
    print(f"SMGCN TCM smoke test passed ({source}); loss={loss.item():.6f}")


if __name__ == "__main__":
    import sys

    _smoke_test(sys.argv[1] if len(sys.argv) > 1 else None)
