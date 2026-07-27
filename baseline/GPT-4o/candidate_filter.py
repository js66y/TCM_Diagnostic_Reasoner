"""
gpt4o_baseline/candidate_filter.py

基于训练集统计，为每个中医诊断预筛选最相关的候选标签：
  - 给定诊断 → Top-K 最常见证型
  - 给定诊断 → Top-K 最常见草药
  - 给定诊断 → Top-K 最常见治法

这样可以把候选集从 266/526 大幅压缩到 30/60，提升 GPT 精准选择的概率。
"""
from __future__ import annotations

import collections
from typing import Optional

import pandas as pd

from utils import (
    COL_TCM_DIAG, COL_SYNDROME, COL_TREATMENT, COL_HERB,
    parse_treatment_ids, parse_herb_ids,
)


class CandidateFilter:
    """
    统计训练集中，每个中医诊断下最常出现的证型/治法/草药，
    用于在推理时缩减候选列表。
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        candidates: dict,
        top_k_syndrome: int = 30,
        top_k_treatment: int = 20,
        top_k_herb: int = 80,
    ):
        self._candidates = candidates
        self.top_k_syndrome = top_k_syndrome
        self.top_k_treatment = top_k_treatment
        self.top_k_herb = top_k_herb

        # {diag_id -> Counter(syndrome_id -> count)}
        self._diag_syndrome: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self._diag_treatment: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self._diag_herb: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

        # 全局频次（当诊断未见过时的 fallback）
        self._global_syndrome: collections.Counter = collections.Counter()
        self._global_treatment: collections.Counter = collections.Counter()
        self._global_herb: collections.Counter = collections.Counter()

        self._build(train_df)
        print(
            f"[CandidateFilter] 统计完毕，覆盖 {len(self._diag_syndrome)} 个诊断类别，"
            f"训练集 {len(train_df)} 条"
        )

    def _build(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                diag_id = int(row[COL_TCM_DIAG]) if pd.notna(row[COL_TCM_DIAG]) else None
            except (ValueError, TypeError):
                diag_id = None

            try:
                syn_id = int(row[COL_SYNDROME]) if pd.notna(row[COL_SYNDROME]) else None
            except (ValueError, TypeError):
                syn_id = None

            treat_ids = parse_treatment_ids(row[COL_TREATMENT])
            herb_ids = parse_herb_ids(row[COL_HERB])

            if syn_id is not None:
                self._global_syndrome[syn_id] += 1
                if diag_id is not None:
                    self._diag_syndrome[diag_id][syn_id] += 1

            for tid in treat_ids:
                self._global_treatment[tid] += 1
                if diag_id is not None:
                    self._diag_treatment[diag_id][tid] += 1

            for hid in herb_ids:
                self._global_herb[hid] += 1
                if diag_id is not None:
                    self._diag_herb[diag_id][hid] += 1

    def _top_names(
        self,
        counter: collections.Counter,
        id2name: dict,
        k: int,
    ) -> list[str]:
        top_ids = [i for i, _ in counter.most_common(k)]
        return [id2name[i] for i in top_ids if i in id2name]

    def filter_candidates(
        self,
        tcm_diag_name: Optional[str] = None,
    ) -> dict:
        """
        返回过滤后的候选字典，结构与 load_candidate_labels() 一致。

        Args:
            tcm_diag_name: GPT 预测的中医诊断名（字符串），None 则使用全局频次。

        Returns:
            过滤后的 candidates dict（只含 syndrome/treatment/herb，tcm_diag 不变）
        """
        name2id_tcm = self._candidates["tcm_diag"]["name2id"]
        diag_id = name2id_tcm.get(tcm_diag_name) if tcm_diag_name else None

        # 选择用 diag-specific 还是 global 统计
        def _pick(diag_counter, global_counter, k):
            if diag_id is not None and diag_id in diag_counter and len(diag_counter[diag_id]) > 0:
                return self._top_names(diag_counter[diag_id], self._candidates["syndrome"]["id2name"]
                                       if diag_counter is self._diag_syndrome else
                                       self._candidates["treatment"]["id2name"]
                                       if diag_counter is self._diag_treatment else
                                       self._candidates["herb"]["id2name"], k)
            return self._top_names(global_counter,
                                   self._candidates["syndrome"]["id2name"]
                                   if global_counter is self._global_syndrome else
                                   self._candidates["treatment"]["id2name"]
                                   if global_counter is self._global_treatment else
                                   self._candidates["herb"]["id2name"], k)

        # 分别取各自的 id2name
        syn_id2name = self._candidates["syndrome"]["id2name"]
        treat_id2name = self._candidates["treatment"]["id2name"]
        herb_id2name = self._candidates["herb"]["id2name"]

        def _top(diag_dict, global_cnt, id2name, k) -> list[str]:
            if diag_id is not None and diag_id in diag_dict and len(diag_dict[diag_id]) > 0:
                src = diag_dict[diag_id]
            else:
                src = global_cnt
            top_ids = [i for i, _ in src.most_common(k)]
            return [id2name[i] for i in top_ids if i in id2name]

        syn_names = _top(self._diag_syndrome, self._global_syndrome, syn_id2name, self.top_k_syndrome)
        treat_names = _top(self._diag_treatment, self._global_treatment, treat_id2name, self.top_k_treatment)
        herb_names = _top(self._diag_herb, self._global_herb, herb_id2name, self.top_k_herb)

        # 构建与原始 candidates 格式相同的结构（只更新 names，name2id/id2name 保持完整以便评估）
        def _sub(names, original):
            n2i = {n: original["name2id"][n] for n in names if n in original["name2id"]}
            i2n = {v: k for k, v in n2i.items()}
            return {"id2name": i2n, "name2id": n2i, "names": names}

        return {
            "tcm_diag": self._candidates["tcm_diag"],  # 不过滤诊断
            "syndrome": _sub(syn_names, self._candidates["syndrome"]),
            "treatment": _sub(treat_names, self._candidates["treatment"]),
            "herb": _sub(herb_names, self._candidates["herb"]),
            "western_diag": self._candidates["western_diag"],
        }
