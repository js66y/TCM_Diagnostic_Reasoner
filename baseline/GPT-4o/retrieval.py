"""
gpt4o_baseline/retrieval.py

基于 TF-IDF 的训练集相似病历检索器（用于动态 RAG few-shot）

使用方式：
    retriever = build_retriever(train_df, candidates)
    examples = retriever.retrieve(query_info, k=5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import row_to_info, COL_CHIEF, COL_HISTORY, COL_PHYSICAL, COL_WESTERN_DIAG


def _build_text(row: pd.Series, western_id2name: dict | None = None) -> str:
    """将病历行拼接为检索用文本，可选加入解码后的西医诊断名"""
    parts = [
        str(row.get(COL_CHIEF, "") or ""),
        str(row.get(COL_HISTORY, "") or ""),
        str(row.get(COL_PHYSICAL, "") or ""),
    ]
    if western_id2name is not None:
        try:
            w_id = int(row.get(COL_WESTERN_DIAG, -1))
            w_name = western_id2name.get(w_id, "")
            if w_name:
                parts.append(w_name)
        except (ValueError, TypeError):
            pass
    return " ".join(p for p in parts if p and p != "nan")


def _build_query_text(info: dict) -> str:
    """将推理信息字典拼接为检索用文本，包含西医初步诊断"""
    parts = [
        info.get("chief_complaint", "") or "",
        info.get("medical_history", "") or "",
        info.get("physical_examination", "") or "",
        info.get("preliminary_western_diagnosis", "") or "",
    ]
    return " ".join(p for p in parts if p)


class CaseRetriever:
    """
    TF-IDF 相似病历检索器。
    - 建立时对训练集文本做 TF-IDF 编码
    - retrieve(query_info, k) 返回最相似的 k 条病历信息字典
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        candidates: dict,
        max_features: int = 8000,
    ):
        self._train_df = train_df.reset_index(drop=True)
        self._candidates = candidates

        # 构建语料（加入西医诊断名，提升对"西医病名明确但中医症状描述模糊"病历的检索精度）
        western_id2name = candidates.get("western_diag", {}).get("id2name", {})
        corpus = [_build_text(row, western_id2name) for _, row in self._train_df.iterrows()]

        # 拟合 TF-IDF（中文按字符级别 n-gram，不依赖分词库）
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)  # (N, vocab)

        # 预计算所有训练样本的 info（用于构造 few-shot）
        self._infos = [
            row_to_info(self._train_df.iloc[i], candidates)
            for i in range(len(self._train_df))
        ]
        print(f"[Retriever] 已建立 TF-IDF 索引，训练集 {len(self._train_df)} 条")

    def retrieve(self, query_info: dict, k: int = 5, exclude_no_herb: bool = True) -> list[dict]:
        """
        检索与 query_info 最相似的 k 条训练样本。

        Args:
            query_info:  row_to_info() 返回的字典
            k:           返回条数
            exclude_no_herb: 排除没有草药标签的训练样本

        Returns:
            list of info dicts，按相似度从高到低排列
        """
        query_text = _build_query_text(query_info)
        query_vec = self._vectorizer.transform([query_text])  # (1, vocab)
        sims = cosine_similarity(query_vec, self._matrix).flatten()  # (N,)

        # 按相似度降序排列，取前 k*3 候选后过滤
        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= k:
                break
            info = self._infos[idx]
            # 过滤：草药列表为空的样本不能作为好的示例
            if exclude_no_herb and not info.get("gt_herb_names"):
                continue
            results.append(info)

        return results
