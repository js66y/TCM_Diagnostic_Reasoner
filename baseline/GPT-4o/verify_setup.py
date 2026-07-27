"""临时验证脚本，确认 utils 和 prompts 能正常工作"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_candidate_labels, get_splits, row_to_info, select_few_shot_examples
from prompts import build_zero_shot_prompt, build_few_shot_prompt

print("=== 1. 加载候选标签 ===")
cands = load_candidate_labels()
for k, v in cands.items():
    print(f"  {k}: {len(v['names'])} 类，前3个: {v['names'][:3]}")

print()
print("=== 2. 数据集分割 ===")
train_df, valid_df, test_df = get_splits()
print(f"  Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

print()
print("=== 3. 第一行 row_to_info ===")
row = test_df.iloc[0]
info = row_to_info(row, cands)
print(f"  诊断: {info['gt_tcm_diag_name']}")
print(f"  证型: {info['gt_syndrome_name']}")
print(f"  治法: {info['gt_treatment_names']}")
print(f"  草药(前5): {info['gt_herb_names'][:5]}")

print()
print("=== 4. Zero-shot prompt 长度 ===")
zs_prompt = build_zero_shot_prompt(info, cands)
print(f"  字符数: {len(zs_prompt)}")
print(f"  估算 token 数: ~{len(zs_prompt)//2}")

print()
print("=== 5. Few-shot 示例选取 ===")
examples = select_few_shot_examples(train_df, cands, n=3)
for i, ex in enumerate(examples):
    print(f"  示例{i+1}: 诊断={ex['gt_tcm_diag_name']}, 证型={ex['gt_syndrome_name']}")

fs_prompt = build_few_shot_prompt(info, cands, examples)
print(f"  Few-shot prompt 字符数: {len(fs_prompt)}")
print(f"  估算 token 数: ~{len(fs_prompt)//2}")

print()
print("=== 全部验证通过 ===")
