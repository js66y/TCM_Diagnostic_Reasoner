"""
gpt4o_baseline/evaluate_gpt4o.py

从 run_gpt4o_baseline.py 生成的 .jsonl 文件计算评估指标，
指标体系与 metrics.py（神经模型）完全一致：

    中医诊断 / 证型：macro-F1（主指标）、accuracy
    治法 / 草药：    micro-F1（主指标）、precision、recall

用法：
    python gpt4o_baseline/evaluate_gpt4o.py --mode zero_shot
    python gpt4o_baseline/evaluate_gpt4o.py --mode few_shot
    python gpt4o_baseline/evaluate_gpt4o.py --mode zero_shot --mode few_shot  # 同时汇报两个
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_candidate_labels

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ────────────────────────────────────────────────────────────────
# 加载预测结果
# ────────────────────────────────────────────────────────────────
def load_predictions(mode: str, tag: str = "") -> list[dict]:
    """
    mode: 如 "rag" / "rag_two_stage"
    tag:  如 "gpt_4o" / "glm_4_7"（对应 --tag 参数，空字符串则尝试不含 tag 的旧格式）
    """
    if tag:
        path = RESULTS_DIR / f"{mode}_{tag}_predictions.jsonl"
    else:
        path = RESULTS_DIR / f"{mode}_predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return records


# ────────────────────────────────────────────────────────────────
# 名称 → ID 映射（用于构造 one-hot 向量）
# ────────────────────────────────────────────────────────────────
def _name_to_id(name: str | None, name2id: dict[str, int]) -> int | None:
    if name is None:
        return None
    return name2id.get(name, None)


def _names_to_multihot(names: list[str], name2id: dict[str, int], n_classes: int) -> np.ndarray:
    vec = np.zeros(n_classes, dtype=int)
    for name in names:
        idx = name2id.get(name)
        if idx is not None:
            vec[idx] = 1
    return vec


# ────────────────────────────────────────────────────────────────
# 评估函数
# ────────────────────────────────────────────────────────────────
def evaluate_multiclass(
    gt_ids: list[int | None],
    pred_ids: list[int | None],
    num_classes: int,
    top_k: int = 3,
) -> dict:
    """单选任务（中医诊断 / 证型）评估"""
    valid = [(g, p) for g, p in zip(gt_ids, pred_ids) if g is not None]
    if not valid:
        return {"n_valid": 0, "accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "score": 0.0}

    gt_arr = np.array([g for g, _ in valid])
    pred_arr = np.array([p if p is not None else -1 for _, p in valid])

    acc = accuracy_score(gt_arr, pred_arr)
    macro_f1 = f1_score(gt_arr, pred_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(gt_arr, pred_arr, average="micro", zero_division=0)

    # none_rate: GPT 返回了候选集外的标签（无法匹配）
    none_count = sum(1 for _, p in valid if p is None)

    return {
        "n_valid": len(valid),
        "n_total": len(gt_ids),
        "none_rate": none_count / len(valid),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "score": float(macro_f1),
    }


def evaluate_multilabel(
    gt_lists: list[list[str]],
    pred_lists: list[list[str]],
    name2id: dict[str, int],
    num_classes: int,
) -> dict:
    """多选任务（治法 / 草药）评估"""
    gt_matrix = np.stack([_names_to_multihot(g, name2id, num_classes) for g in gt_lists])
    pred_matrix = np.stack([_names_to_multihot(p, name2id, num_classes) for p in pred_lists])

    micro_f1 = f1_score(gt_matrix, pred_matrix, average="micro", zero_division=0)
    macro_f1 = f1_score(gt_matrix, pred_matrix, average="macro", zero_division=0)
    precision = precision_score(gt_matrix, pred_matrix, average="micro", zero_division=0)
    recall = recall_score(gt_matrix, pred_matrix, average="micro", zero_division=0)

    # 平均每条预测数 & 真实标签数
    pred_count_avg = float(pred_matrix.sum(axis=1).mean())
    gt_count_avg = float(gt_matrix.sum(axis=1).mean())

    return {
        "n_samples": len(gt_lists),
        "gt_avg_labels": round(gt_count_avg, 2),
        "pred_avg_labels": round(pred_count_avg, 2),
        "precision": float(precision),
        "recall": float(recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "score": float(micro_f1),
    }


# ────────────────────────────────────────────────────────────────
# 主评估入口
# ────────────────────────────────────────────────────────────────
def evaluate_mode(mode: str, candidates: dict, tag: str = "") -> dict:
    records = load_predictions(mode, tag=tag)

    tcm_n2id = candidates["tcm_diag"]["name2id"]
    syn_n2id = candidates["syndrome"]["name2id"]
    treat_n2id = candidates["treatment"]["name2id"]
    herb_n2id = candidates["herb"]["name2id"]

    n_tcm = len(tcm_n2id)
    n_syn = len(syn_n2id)
    n_treat = len(treat_n2id)
    n_herb = len(herb_n2id)

    gt_tcm_ids, pred_tcm_ids = [], []
    gt_syn_ids, pred_syn_ids = [], []
    gt_treat_lists, pred_treat_lists = [], []
    gt_herb_lists, pred_herb_lists = [], []

    for rec in records:
        gt = rec["gt"]
        pred = rec["pred"]

        gt_tcm_ids.append(tcm_n2id.get(gt["tcm_diag_name"]))
        pred_tcm_ids.append(_name_to_id(pred.get("tcm_diagnosis"), tcm_n2id))

        gt_syn_ids.append(syn_n2id.get(gt["syndrome_name"]))
        pred_syn_ids.append(_name_to_id(pred.get("syndrome_type"), syn_n2id))

        gt_treat_lists.append(gt["treatment_names"])
        pred_treat_lists.append(pred.get("treatment_method", []))

        gt_herb_lists.append(gt["herb_names"])
        pred_herb_lists.append(pred.get("herb_recommendation", []))

    diag_metrics = evaluate_multiclass(gt_tcm_ids, pred_tcm_ids, n_tcm)
    syn_metrics = evaluate_multiclass(gt_syn_ids, pred_syn_ids, n_syn)
    treat_metrics = evaluate_multilabel(gt_treat_lists, pred_treat_lists, treat_n2id, n_treat)
    herb_metrics = evaluate_multilabel(gt_herb_lists, pred_herb_lists, herb_n2id, n_herb)

    chain_score = (
        diag_metrics["score"]
        + syn_metrics["score"]
        + treat_metrics["score"]
        + herb_metrics["score"]
    ) / 4.0

    return {
        "mode": mode,
        "n_samples": len(records),
        "chain_score": round(chain_score, 4),
        "tcm_diagnosis": diag_metrics,
        "syndrome": syn_metrics,
        "treatment": treat_metrics,
        "herb": herb_metrics,
    }


def print_report(result: dict):
    mode = result["mode"]
    n = result["n_samples"]
    print(f"\n{'='*60}")
    print(f"  基线评估报告 [{mode}]")
    print(f"{'='*60}")
    print(f"  Chain Score (avg): {result['chain_score']:.4f}\n")

    # 跳过纯计数字段，只打印指标
    _skip = {"n_valid", "n_total", "n_samples"}

    for task_key, label in [
        ("tcm_diagnosis", "中医诊断"),
        ("syndrome", "证型"),
        ("treatment", "治法"),
        ("herb", "草药"),
    ]:
        m = result[task_key]
        metrics = {k: v for k, v in m.items() if k not in _skip}
        print(f"  [{label}]")
        for k, v in metrics.items():
            print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
        print()


def main():
    parser = argparse.ArgumentParser(description="基线评估")
    parser.add_argument(
        "--mode",
        choices=[
            "zero_shot", "few_shot", "rag",
            "zero_shot_two_stage", "few_shot_two_stage", "rag_two_stage",
            "zero_shot_four_stage", "few_shot_four_stage", "rag_four_stage",
        ],
        nargs="+",
        default=["zero_shot"],
        help="评估模式（可同时指定多个，如 --mode rag rag_four_stage）",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="与 run_gpt4o_baseline.py --tag 对应，用于定位多模型结果文件（空=旧格式无tag）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="将评估结果保存为 results/metrics_report.json",
    )
    args = parser.parse_args()

    candidates = load_candidate_labels()
    all_results = {}

    for mode in args.mode:
        try:
            result = evaluate_mode(mode, candidates, tag=args.tag)
            print_report(result)
            key = f"{mode}_{args.tag}" if args.tag else mode
            all_results[key] = result
        except FileNotFoundError as e:
            print(f"[跳过 {mode}] {e}")

    if args.save and all_results:
        out = RESULTS_DIR / "metrics_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n评估报告已保存至 {out}")


if __name__ == "__main__":
    main()
