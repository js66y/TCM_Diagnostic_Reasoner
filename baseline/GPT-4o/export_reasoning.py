"""
gpt4o_baseline/export_reasoning.py

将预测结果 JSONL 导出为可读 CSV，方便逐条检查推理过程。

用法：
    python gpt4o_baseline/export_reasoning.py --mode rag
    python gpt4o_baseline/export_reasoning.py --mode rag --tag glm_4_7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_jsonl(mode: str, tag: str = "") -> list[dict]:
    if tag:
        path = RESULTS_DIR / f"{mode}_{tag}_predictions.jsonl"
    else:
        path = RESULTS_DIR / f"{mode}_predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return records


def is_correct_single(gt: str | None, pred: str | None) -> str:
    if gt is None or pred is None:
        return "N/A"
    return "✓" if gt == pred else "✗"


def list_overlap(gt: list, pred: list) -> str:
    """显示交集/召回/精准"""
    gt_set = set(gt)
    pred_set = set(pred)
    hit = gt_set & pred_set
    if not gt_set:
        return "N/A"
    p = len(hit) / len(pred_set) if pred_set else 0
    r = len(hit) / len(gt_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return f"F1={f1:.2f} | 命中{len(hit)}/{len(gt_set)}个GT | 预测{len(pred_set)}个"


def export(mode: str, tag: str = ""):
    records = load_jsonl(mode, tag)
    records.sort(key=lambda r: r.get("row_idx", 0))

    suffix = f"_{tag}" if tag else ""
    out_path = RESULTS_DIR / f"{mode}{suffix}_reasoning.csv"

    fieldnames = [
        "序号",
        # 真实标签
        "GT_中医诊断", "GT_证型", "GT_治法", "GT_草药",
        # 预测结果
        "PRED_中医诊断", "PRED_证型", "PRED_治法", "PRED_草药",
        # 对错标记
        "诊断对否", "证型对否", "治法评估", "草药评估",
        # 推理过程
        "模型推理过程",
        # 原始输出（如有错误可以查原文）
        "PRED_中医诊断_RAW", "PRED_证型_RAW",
    ]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            gt = rec.get("gt", {})
            pred = rec.get("pred", {})

            # 治法/草药 GT 可能是 list，转成文字
            gt_treat = "、".join(gt.get("treatment_names", []))
            gt_herb  = "、".join(gt.get("herb_names", []))
            pred_treat = "、".join(pred.get("treatment_method", []))
            pred_herb  = "、".join(pred.get("herb_recommendation", []))

            writer.writerow({
                "序号": rec.get("row_idx", ""),
                "GT_中医诊断": gt.get("tcm_diag_name", ""),
                "GT_证型":    gt.get("syndrome_name", ""),
                "GT_治法":    gt_treat,
                "GT_草药":    gt_herb,
                "PRED_中医诊断": pred.get("tcm_diagnosis", "") or "",
                "PRED_证型":    pred.get("syndrome_type", "") or "",
                "PRED_治法":    pred_treat,
                "PRED_草药":    pred_herb,
                "诊断对否": is_correct_single(
                    gt.get("tcm_diag_name"), pred.get("tcm_diagnosis")
                ),
                "证型对否": is_correct_single(
                    gt.get("syndrome_name"), pred.get("syndrome_type")
                ),
                "治法评估": list_overlap(
                    gt.get("treatment_names", []),
                    pred.get("treatment_method", []),
                ),
                "草药评估": list_overlap(
                    gt.get("herb_names", []),
                    pred.get("herb_recommendation", []),
                ),
                "模型推理过程": pred.get("reasoning", ""),
                "PRED_中医诊断_RAW": pred.get("tcm_diagnosis_raw", "") or "",
                "PRED_证型_RAW":    pred.get("syndrome_type_raw", "") or "",
            })

    print(f"已导出 {len(records)} 条 → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="导出推理过程到 CSV")
    parser.add_argument("--mode", default="rag",
                        choices=["zero_shot", "few_shot", "rag",
                                 "zero_shot_two_stage", "few_shot_two_stage", "rag_two_stage"])
    parser.add_argument("--tag", default="",
                        help="与 run_gpt4o_baseline.py --tag 对应（空=无tag）")
    args = parser.parse_args()
    export(args.mode, args.tag)


if __name__ == "__main__":
    main()
