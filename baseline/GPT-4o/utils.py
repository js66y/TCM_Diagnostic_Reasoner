"""
gpt4o_baseline/utils.py
数据加载、标签映射、数据集分割复现
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────────────────────
# 路径常量（相对于项目根目录）
# ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "clear888-add-with-evidence-indexed.csv"
TCM_DIAG_MAP_PATH   = ROOT / "Dataset" / "中医诊断.csv"
SYNDROME_MAP_PATH   = ROOT / "Dataset" / "中医证型.csv"
TREATMENT_MAP_PATH  = ROOT / "Dataset" / "中医治则_frequency_rank.csv"
HERB_MAP_PATH       = ROOT / "Dataset" / "处方内容.csv"
WESTERN_DIAG_MAP_PATH = ROOT / "Dataset" / "诊断.csv"
WESTERN_DIAG_MAP_PATH = ROOT / "Dataset" / "诊断.csv"

# 主数据集列名（与 main.py 保持一致）
COL_SEQ = "序号"
COL_CASE = "病历号"
COL_GENDER = "性别"
COL_AGE = "年龄"
COL_WESTERN_DIAG = "初步诊断"
COL_TCM_DIAG = "中医诊断"
COL_SYNDROME = "证型"
COL_TREATMENT = "治则治法"
COL_HERB = "药名与单帖重量"
COL_CHIEF = "主诉"
COL_HISTORY = "简要病史"
COL_PHYSICAL = "体格检查"
COL_SUMMARY = "病历汇总"
COL_TOKENS = "病历文本词源"

RANDOM_SEED = 42


# ────────────────────────────────────────────────────────────────
# 编码兼容读取
# ────────────────────────────────────────────────────────────────
def _read_csv(path: str | Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gbk", "gb18030", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            pass
    raise ValueError(f"无法读取文件：{path}")


# ────────────────────────────────────────────────────────────────
# 候选标签加载
# ────────────────────────────────────────────────────────────────
def load_candidate_labels() -> dict[str, dict]:
    """
    返回:
        {
          "tcm_diag":     {"id2name": {0: "胃痞病", ...}, "name2id": {...}, "names": [...]},
          "syndrome":     {...},
          "treatment":    {...},
          "herb":         {...},
          "western_diag": {...},   # 初步诊断映射（仅用于病历展示，不作为候选标签）
        }
    """
    def _build(df: pd.DataFrame, name_col: int, id_col: int, clean_fn=None):
        id2name: dict[int, str] = {}
        name2id: dict[str, int] = {}
        for _, row in df.iterrows():
            raw_name = str(row.iloc[name_col]).strip()
            name = clean_fn(raw_name) if clean_fn else raw_name
            try:
                idx = int(row.iloc[id_col])
            except (ValueError, TypeError):
                continue
            id2name[idx] = name
            name2id[name] = idx
        names = [id2name[k] for k in sorted(id2name)]
        return {"id2name": id2name, "name2id": name2id, "names": names}

    def _clean_western(name: str) -> str:
        # 诊断.csv 格式为 "112342|高血压"，取竖线后半段
        return name.split("|")[-1].strip() if "|" in name else name

    tcm_diag_df   = _read_csv(TCM_DIAG_MAP_PATH)      # col0=病名,     col1=编号
    syndrome_df   = _read_csv(SYNDROME_MAP_PATH)       # col0=证型,     col1=编号
    herb_df       = _read_csv(HERB_MAP_PATH)           # col0=中药名称, col1=编号
    treatment_df  = _read_csv(TREATMENT_MAP_PATH)      # col0=Word,     col2=ID
    western_df    = _read_csv(WESTERN_DIAG_MAP_PATH)   # col0=诊断名,   col1=编号

    return {
        "tcm_diag":     _build(tcm_diag_df,  name_col=0, id_col=1),
        "syndrome":     _build(syndrome_df,   name_col=0, id_col=1),
        "herb":         _build(herb_df,       name_col=0, id_col=1),
        "treatment":    _build(treatment_df,  name_col=0, id_col=2),
        "western_diag": _build(western_df,    name_col=0, id_col=1, clean_fn=_clean_western),
    }


# ────────────────────────────────────────────────────────────────
# 数据集分割（完全复现 main.py 的逻辑）
# ────────────────────────────────────────────────────────────────
def get_splits(data_path: str | Path = DATA_PATH):
    """
    复现 main.py 中的 train/valid/test 分割：
        train_test_split(df, test_size=0.2, random_state=42)
        train_test_split(test_df, test_size=0.5, random_state=42)
    返回 (train_df, valid_df, test_df)
    """
    df = _read_csv(data_path)
    # 用列索引重命名，保证中文列名在不同环境中一致
    col_map = {
        df.columns[0]: COL_SEQ,
        df.columns[1]: COL_CASE,
        df.columns[2]: COL_GENDER,
        df.columns[3]: COL_AGE,
        df.columns[4]: COL_WESTERN_DIAG,
        df.columns[5]: COL_TCM_DIAG,
        df.columns[6]: COL_SYNDROME,
        df.columns[7]: COL_TREATMENT,
        df.columns[8]: COL_HERB,
        df.columns[9]: COL_CHIEF,
        df.columns[10]: COL_HISTORY,
        df.columns[11]: COL_PHYSICAL,
        df.columns[12]: COL_SUMMARY,
        df.columns[13]: COL_TOKENS,
    }
    df = df.rename(columns=col_map)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=RANDOM_SEED)
    print(f"数据集分割: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────
# 字段解析工具
# ────────────────────────────────────────────────────────────────
def parse_treatment_ids(s) -> list[int]:
    """解析治则治法列（存储为列表字符串 '[23, 92]'）"""
    if not isinstance(s, str):
        return []
    try:
        result = ast.literal_eval(s)
        return [int(x) for x in result]
    except (ValueError, SyntaxError):
        return []


def parse_herb_ids(s) -> list[int]:
    """解析草药列（存储为字典字符串 '{2: 12.0, 3: 6.0}'）"""
    if not isinstance(s, str):
        return []
    try:
        d = ast.literal_eval(s)
        return [int(k) for k in d.keys()]
    except (ValueError, SyntaxError):
        return []


def row_to_info(row: pd.Series, candidates: dict) -> dict:
    """
    将一行 DataFrame 转换为推理所需的信息字典：
    包含文本字段 + 解析后的标签名称（用于 few-shot 示例展示）
    """
    tcm_diag_id = int(row[COL_TCM_DIAG]) if pd.notna(row[COL_TCM_DIAG]) else -1
    syndrome_id = int(row[COL_SYNDROME]) if pd.notna(row[COL_SYNDROME]) else -1
    treatment_ids = parse_treatment_ids(row[COL_TREATMENT])
    herb_ids = parse_herb_ids(row[COL_HERB])

    return {
        "gender": "男" if int(row[COL_GENDER]) == 0 else "女",
        "age": int(row[COL_AGE]),
        "chief_complaint": str(row[COL_CHIEF]) if pd.notna(row[COL_CHIEF]) else "",
        "medical_history": str(row[COL_HISTORY]) if pd.notna(row[COL_HISTORY]) else "",
        "physical_examination": str(row[COL_PHYSICAL]) if pd.notna(row[COL_PHYSICAL]) else "",
        "preliminary_western_diagnosis": candidates["western_diag"]["id2name"].get(
            int(row[COL_WESTERN_DIAG]) if pd.notna(row[COL_WESTERN_DIAG]) else -1, ""
        ),
        # 真实标签（评估用 / few-shot 示例用）
        "gt_tcm_diag_id": tcm_diag_id,
        "gt_tcm_diag_name": candidates["tcm_diag"]["id2name"].get(tcm_diag_id, ""),
        "gt_syndrome_id": syndrome_id,
        "gt_syndrome_name": candidates["syndrome"]["id2name"].get(syndrome_id, ""),
        "gt_treatment_ids": treatment_ids,
        "gt_treatment_names": [candidates["treatment"]["id2name"].get(i, "") for i in treatment_ids],
        "gt_herb_ids": herb_ids,
        "gt_herb_names": [candidates["herb"]["id2name"].get(i, "") for i in herb_ids],
    }


# ────────────────────────────────────────────────────────────────
# Few-shot 示例选取（固定 3 条，覆盖高频诊断）
# ────────────────────────────────────────────────────────────────
def select_few_shot_examples(train_df: pd.DataFrame, candidates: dict, n: int = 3) -> list[dict]:
    """
    从训练集中选取 n 条固定示例，策略：按中医诊断频次取前 n 个最常见诊断，
    每个诊断取一条代表样本，保证多样性且跨运行稳定。
    """
    top_diag_ids = (
        train_df[COL_TCM_DIAG]
        .dropna()
        .astype(int)
        .value_counts()
        .head(n * 3)
        .index.tolist()
    )
    examples = []
    used_diags = set()
    for diag_id in top_diag_ids:
        if len(examples) >= n:
            break
        if diag_id in used_diags:
            continue
        subset = train_df[train_df[COL_TCM_DIAG].astype(float).astype(int) == diag_id]
        if subset.empty:
            continue
        row = subset.iloc[0]
        info = row_to_info(row, candidates)
        examples.append(info)
        used_diags.add(diag_id)
    return examples
