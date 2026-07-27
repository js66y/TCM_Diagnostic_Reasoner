"""分析GT标签分布和Prompt规模"""
import sys
sys.path.insert(0, "gpt4o_baseline")
from utils import load_candidate_labels, get_splits, COL_SYNDROME, COL_TCM_DIAG

c = load_candidate_labels()
_, _, test_df = get_splits()

syn_dist = test_df[COL_SYNDROME].dropna().astype(int).value_counts()
tcm_dist = test_df[COL_TCM_DIAG].dropna().astype(int).value_counts()

n_syn_candidates = len(c["syndrome"]["names"])
n_tcm_candidates = len(c["tcm_diag"]["names"])
n_herb_candidates = len(c["herb"]["names"])
n_treat_candidates = len(c["treatment"]["names"])

print("=== Candidate Set Size ===")
print(f"  TCM Diagnosis candidates: {n_tcm_candidates}")
print(f"  Syndrome candidates:      {n_syn_candidates}")
print(f"  Treatment candidates:     {n_treat_candidates}")
print(f"  Herb candidates:          {n_herb_candidates}")
print(f"  Test set size:            {len(test_df)}")

print()
print("=== GT Syndrome Distribution in Test Set ===")
print(f"  Unique syndromes in test: {len(syn_dist)} / {n_syn_candidates} candidates")
print(f"  Top-5 syndromes cover:    {syn_dist.head(5).sum()/len(test_df)*100:.1f}% of test")
print(f"  Top-10 syndromes cover:   {syn_dist.head(10).sum()/len(test_df)*100:.1f}% of test")
print(f"  Syndromes seen >= 10x:    {(syn_dist >= 10).sum()}")
print(f"  Syndromes seen only 1x:   {(syn_dist == 1).sum()}")
top5_syn = [c["syndrome"]["id2name"].get(int(i), "?") for i in syn_dist.head(5).index]
print(f"  Top-5 names: {top5_syn}")

print()
print("=== GT TCM Diagnosis Distribution in Test Set ===")
print(f"  Unique diagnoses in test: {len(tcm_dist)} / {n_tcm_candidates} candidates")
print(f"  Top-5 diagnoses cover:    {tcm_dist.head(5).sum()/len(test_df)*100:.1f}% of test")
top5_tcm = [c["tcm_diag"]["id2name"].get(int(i), "?") for i in tcm_dist.head(5).index]
print(f"  Top-5 names: {top5_tcm}")

print()
print("=== Prompt Candidate List Length (chars) ===")
syn_str = "\u3001".join(c["syndrome"]["names"])
herb_str = "\u3001".join(c["herb"]["names"])
treat_str = "\u3001".join(c["treatment"]["names"])
tcm_str = "\u3001".join(c["tcm_diag"]["names"])
total = len(syn_str) + len(herb_str) + len(treat_str) + len(tcm_str)
print(f"  Syndrome list:   {len(syn_str)} chars")
print(f"  Herb list:       {len(herb_str)} chars")
print(f"  Treatment list:  {len(treat_str)} chars")
print(f"  TCM diag list:   {len(tcm_str)} chars")
print(f"  Total candidate: {total} chars (~{total//4} tokens, very long!)")
