"""诊断预测结果质量的脚本"""
import json
import sys
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_candidate_labels, get_splits

# ── 候选标签规模 ──────────────────────────────────────────────
c = load_candidate_labels()
print("=== 候选标签规模 ===")
print(f"  中医诊断: {len(c['tcm_diag']['names'])} 类")
print(f"  证型:     {len(c['syndrome']['names'])} 类")
print(f"  治法:     {len(c['treatment']['names'])} 类")
print(f"  草药:     {len(c['herb']['names'])} 类")

_, _, test_df = get_splits()
print(f"\n测试集大小: {len(test_df)}")

# ── 证型候选前30 ──────────────────────────────────────────────
print("\n证型候选（前30）:")
for name in c["syndrome"]["names"][:30]:
    print(f"  {name}")

# ── 加载预测 ──────────────────────────────────────────────────
path = Path(__file__).parent / "results" / "few_shot_predictions.jsonl"
if not path.exists():
    print("\n预测文件不存在，跳过预测分析")
    sys.exit(0)

records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
print(f"\n=== 预测文件统计（few_shot, n={len(records)}）===")

parse_errors = sum(1 for r in records if "parse_error" in r["pred"])
syn_none = sum(1 for r in records if r["pred"].get("syndrome_type") is None)
tcm_none = sum(1 for r in records if r["pred"].get("tcm_diagnosis") is None)
treat_empty = sum(1 for r in records if not r["pred"].get("treatment_method"))

print(f"  parse_error 条数:              {parse_errors} ({parse_errors/len(records)*100:.1f}%)")
print(f"  tcm_diagnosis 不在候选集(None): {tcm_none} ({tcm_none/len(records)*100:.1f}%)")
print(f"  syndrome_type 不在候选集(None): {syn_none} ({syn_none/len(records)*100:.1f}%)")
print(f"  treatment_method 为空:          {treat_empty} ({treat_empty/len(records)*100:.1f}%)")

herb_counts = [len(r["pred"].get("herb_recommendation", [])) for r in records]
print(f"\n草药数量分布 (min={min(herb_counts)}, max={max(herb_counts)}, avg={sum(herb_counts)/len(herb_counts):.1f}):")
dist = collections.Counter(herb_counts)
for k in sorted(dist):
    print(f"  {k:>2}味: {dist[k]}条")

# ── 前5条对比 ──────────────────────────────────────────────────
print("\n=== 前5条 syndrome 原始输出 vs GT ===")
for r in records[:5]:
    gt_syn = r["gt"]["syndrome_name"]
    pred_syn = r["pred"].get("syndrome_type")
    raw_syn = r["pred"].get("syndrome_type_raw")
    match = "OK" if pred_syn == gt_syn else "XX"
    print(f"  [{match}] GT:   {gt_syn}")
    print(f"         PRED: {pred_syn}  (raw: {raw_syn})")
    print()

# ── 分析syndrome_raw 里最常出现的错误值 ──────────────────────
print("=== syndrome_type_raw 最常见的20种输出 ===")
raw_counter = collections.Counter()
for r in records:
    raw = r["pred"].get("syndrome_type_raw")
    if raw:
        raw_counter[raw] += 1
for val, cnt in raw_counter.most_common(20):
    in_set = val in c["syndrome"]["name2id"]
    flag = "[OK]" if in_set else "[XX-not in candidates]"
    print(f"  [{cnt:>3}次] {flag}  {val}")

# ── tcm_diagnosis 最常见错误 ──────────────────────────────────
print("\n=== tcm_diagnosis_raw 最常见的20种输出 ===")
tcm_counter = collections.Counter()
for r in records:
    raw = r["pred"].get("tcm_diagnosis_raw")
    if raw:
        tcm_counter[raw] += 1
for val, cnt in tcm_counter.most_common(20):
    in_set = val in c["tcm_diag"]["name2id"]
    flag = "[OK]" if in_set else "[XX-not in candidates]"
    print(f"  [{cnt:>3}次] {flag}  {val}")
