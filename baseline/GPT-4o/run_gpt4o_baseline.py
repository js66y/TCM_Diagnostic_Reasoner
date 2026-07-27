"""
gpt4o_baseline/run_gpt4o_baseline.py

GPT-4o 中医诊疗链推理基线
支持三种模式：zero_shot / few_shot / rag
支持断点续传，支持 --dry_run N 只跑前 N 条

用法示例：
    python gpt4o_baseline/run_gpt4o_baseline.py --mode zero_shot --dry_run 100
    python gpt4o_baseline/run_gpt4o_baseline.py --mode few_shot
    python gpt4o_baseline/run_gpt4o_baseline.py --mode rag --rag_k 5 --filter_candidates

环境变量（必须设置）：
    OPENAI_API_KEY=sk-xxx
    OPENAI_BASE_URL=https://api.openai.com/v1   # 可选，默认官方
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from tqdm import tqdm

# 将项目根目录加入 path，以便 import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    get_splits,
    load_candidate_labels,
    row_to_info,
    select_few_shot_examples,
)
from prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_STAGE1,
    SYSTEM_PROMPT_STAGE2,
    SYSTEM_PROMPT_STEP2,
    SYSTEM_PROMPT_STEP3,
    SYSTEM_PROMPT_STEP4,
    build_few_shot_prompt,
    build_zero_shot_prompt,
    build_rag_prompt,
    build_stage1_prompt,
    build_stage2_prompt,
    build_stage2_rag_prompt,
    build_step2_prompt,
    build_step3_prompt,
    build_step4_prompt,
    build_step1_rag_prompt,
    build_step2_rag_prompt,
    build_step3_rag_prompt,
    build_step4_rag_prompt,
)
from retrieval import CaseRetriever
from candidate_filter import CandidateFilter

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────
# OpenAI 客户端初始化
# ────────────────────────────────────────────────────────────────
def _init_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "未找到 OPENAI_API_KEY 环境变量。\n"
            "请在项目根目录新建 .env 文件，写入：\n"
            "  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx\n"
            "然后执行：\n"
            "  $env:OPENAI_API_KEY='sk-xxxxxx'   # PowerShell\n"
            "  export OPENAI_API_KEY=sk-xxxxxx   # bash/zsh"
        )
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    return openai.OpenAI(api_key=api_key, base_url=base_url)


# ────────────────────────────────────────────────────────────────
# GPT-4o 调用（含重试）
# ────────────────────────────────────────────────────────────────
def _call_gpt4o(
    client: openai.OpenAI,
    user_prompt: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
    retry_delay: float = 5.0,
    system_prompt: str | None = None,
    no_json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """调用模型 API，失败时自动重试，返回原始文本"""
    sys_msg = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not no_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except openai.RateLimitError:
            wait = retry_delay * (2 ** attempt)
            print(f"\n[RateLimit] 第 {attempt+1}/{max_retries} 次，等待 {wait:.0f}s ...")
            time.sleep(wait)
        except openai.APIError as e:
            print(f"\n[APIError] {e}，第 {attempt+1}/{max_retries} 次重试 ...")
            time.sleep(retry_delay)
    raise RuntimeError("API 调用失败，已超过最大重试次数")


# ────────────────────────────────────────────────────────────────
# JSON 解析（健壮版）
# ────────────────────────────────────────────────────────────────
def _parse_response(raw: str, candidates: dict) -> dict:
    """
    解析 GPT-4o 返回的 JSON，对标签做合法性校验：
    - 单选字段：若不在候选集中，记为 None
    - 多选字段：过滤掉不在候选集中的项
    """
    try:
        # 去掉可能的 markdown 代码块包裹
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_error": raw,
            "tcm_diagnosis": None,
            "syndrome_type": None,
            "treatment_method": [],
            "herb_recommendation": [],
            "reasoning": "",
        }

    tcm_name_set = set(candidates["tcm_diag"]["names"])
    syn_name_set = set(candidates["syndrome"]["names"])
    treat_name_set = set(candidates["treatment"]["names"])
    herb_name_set = set(candidates["herb"]["names"])

    tcm_diag = data.get("tcm_diagnosis")
    syndrome = data.get("syndrome_type")
    treatment = data.get("treatment_method", [])
    herb = data.get("herb_recommendation", [])

    return {
        "reasoning": data.get("reasoning", ""),
        "tcm_diagnosis": tcm_diag if tcm_diag in tcm_name_set else None,
        "syndrome_type": syndrome if syndrome in syn_name_set else None,
        "treatment_method": [t for t in (treatment or []) if t in treat_name_set],
        "herb_recommendation": [h for h in (herb or []) if h in herb_name_set],
        "tcm_diagnosis_raw": tcm_diag,
        "syndrome_type_raw": syndrome,
        "treatment_method_raw": treatment,
        "herb_recommendation_raw": herb,
    }


# ────────────────────────────────────────────────────────────────
# 两阶段推理解析函数
# ────────────────────────────────────────────────────────────────
def _parse_stage1_response(raw: str, candidates: dict) -> dict:
    """解析第一阶段（仅中医诊断）的 GPT 返回"""
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"tcm_diagnosis": None, "tcm_diagnosis_raw": None, "reasoning": ""}

    tcm_name_set = set(candidates["tcm_diag"]["names"])
    tcm_diag = data.get("tcm_diagnosis")
    return {
        "reasoning": data.get("reasoning", ""),
        "tcm_diagnosis": tcm_diag if tcm_diag in tcm_name_set else None,
        "tcm_diagnosis_raw": tcm_diag,
    }


def _parse_stage2_response(raw: str, candidates: dict) -> dict:
    """解析第二阶段（证型/治法/草药）的 GPT 返回"""
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_error": raw,
            "syndrome_type": None,
            "treatment_method": [],
            "herb_recommendation": [],
            "reasoning": "",
        }

    syn_name_set = set(candidates["syndrome"]["names"])
    treat_name_set = set(candidates["treatment"]["names"])
    herb_name_set = set(candidates["herb"]["names"])

    syndrome = data.get("syndrome_type")
    treatment = data.get("treatment_method", [])
    herb = data.get("herb_recommendation", [])

    return {
        "reasoning": data.get("reasoning", ""),
        "syndrome_type": syndrome if syndrome in syn_name_set else None,
        "treatment_method": [t for t in (treatment or []) if t in treat_name_set],
        "herb_recommendation": [h for h in (herb or []) if h in herb_name_set],
        "syndrome_type_raw": syndrome,
        "treatment_method_raw": treatment,
        "herb_recommendation_raw": herb,
    }


# ────────────────────────────────────────────────────────────────
# 四步链式推理解析函数
# ────────────────────────────────────────────────────────────────
def _parse_step2_response(raw: str, candidates: dict) -> dict:
    """解析 Step2（仅证型）的 GPT 返回"""
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"syndrome_type": None, "syndrome_type_raw": None, "reasoning": ""}

    syn_name_set = set(candidates["syndrome"]["names"])
    syndrome = data.get("syndrome_type")
    return {
        "reasoning": data.get("reasoning", ""),
        "syndrome_type": syndrome if syndrome in syn_name_set else None,
        "syndrome_type_raw": syndrome,
    }


def _parse_step3_response(raw: str, candidates: dict) -> dict:
    """解析 Step3（仅治法）的 GPT 返回"""
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"treatment_method": [], "treatment_method_raw": [], "reasoning": ""}

    treat_name_set = set(candidates["treatment"]["names"])
    treatment = data.get("treatment_method", [])
    return {
        "reasoning": data.get("reasoning", ""),
        "treatment_method": [t for t in (treatment or []) if t in treat_name_set],
        "treatment_method_raw": treatment,
    }


def _parse_step4_response(raw: str, candidates: dict) -> dict:
    """解析 Step4（仅草药）的 GPT 返回"""
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"herb_recommendation": [], "herb_recommendation_raw": [], "reasoning": ""}

    herb_name_set = set(candidates["herb"]["names"])
    herb = data.get("herb_recommendation", [])
    return {
        "reasoning": data.get("reasoning", ""),
        "herb_recommendation": [h for h in (herb or []) if h in herb_name_set],
        "herb_recommendation_raw": herb,
    }


# ────────────────────────────────────────────────────────────────
# 断点续传
# ────────────────────────────────────────────────────────────────
def _load_done_ids(output_path: Path) -> set[int]:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["row_idx"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


_write_lock = threading.Lock()


def _append_result(output_path: Path, record: dict):
    with _write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────
def run(args):
    # 1. 初始化 OpenAI 客户端
    client = _init_client()

    # 2. 加载数据
    print("加载候选标签...")
    candidates = load_candidate_labels()
    print(
        f"  中医诊断: {len(candidates['tcm_diag']['names'])} 类  "
        f"证型: {len(candidates['syndrome']['names'])} 类  "
        f"治法: {len(candidates['treatment']['names'])} 类  "
        f"草药: {len(candidates['herb']['names'])} 类"
    )

    print("加载数据集...")
    train_df, _, test_df = get_splits()

    # 3. 模式特定初始化
    few_shot_examples = []
    retriever = None
    cand_filter = None

    if args.mode == "few_shot":
        few_shot_examples = select_few_shot_examples(train_df, candidates, n=args.few_shot_k)
        print(f"Few-shot 示例诊断：{[ex['gt_tcm_diag_name'] for ex in few_shot_examples]}")

    elif args.mode == "rag":
        print("构建 TF-IDF 检索索引（首次运行约需 30 秒）...")
        retriever = CaseRetriever(train_df, candidates)

    if args.filter_candidates or args.two_stage or args.four_stage:
        print("构建候选标签过滤器（统计训练集标签频次）...")
        cand_filter = CandidateFilter(
            train_df, candidates,
            top_k_syndrome=args.filter_syn_k,
            top_k_treatment=args.filter_treat_k,
            top_k_herb=args.filter_herb_k,
        )
        print(
            f"  候选集已压缩：证型 {len(candidates['syndrome']['names'])}→{args.filter_syn_k}，"
            f"草药 {len(candidates['herb']['names'])}→{args.filter_herb_k}"
        )

    # 4. 确定输出文件
    if args.four_stage:
        suffix = "_four_stage"
    elif args.two_stage:
        suffix = "_two_stage"
    else:
        suffix = ""
    # tag 默认用模型名（去掉特殊字符），方便多模型并跑时区分文件
    tag = args.tag if args.tag else re.sub(r"[^\w]", "_", args.model)
    output_path = RESULTS_DIR / f"{args.mode}{suffix}_{tag}_predictions.jsonl"
    done_ids = _load_done_ids(output_path)
    print(f"输出文件: {output_path}")
    print(f"已完成: {len(done_ids)} 条，续接运行...")

    # 5. 确定待处理范围
    total = min(args.dry_run, len(test_df)) if args.dry_run > 0 else len(test_df)
    indices = [i for i in range(total) if i not in done_ids]
    print(f"本次需处理: {len(indices)} 条 / 共 {total} 条")

    # 6. 并发推理
    def _process_one(i: int) -> dict:
        import collections as _collections
        row = test_df.iloc[i]
        info = row_to_info(row, candidates)

        # ── 四步链式推理路径 ────────────────────────────────────────
        if args.four_stage:
            # RAG 模式：提前检索相似病例，供后续各步复用
            retrieved = retriever.retrieve(info, k=args.rag_k) if args.mode == "rag" else []

            # Step 1：预测中医诊断（RAG 模式注入检索摘要）
            if args.mode == "rag" and retrieved:
                s1_prompt = build_step1_rag_prompt(info, candidates, retrieved)
            else:
                s1_prompt = build_stage1_prompt(info, candidates)
            try:
                raw1 = _call_gpt4o(
                    client, s1_prompt, model=args.model,
                    system_prompt=SYSTEM_PROMPT_STAGE1,
                    no_json_mode=args.no_json_mode,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except RuntimeError as e:
                print(f"\n[SKIP row {i} step1] {e}")
                raw1 = ""
            s1 = _parse_stage1_response(raw1, candidates)
            predicted_diag = s1["tcm_diagnosis"]

            # 按诊断精筛候选集（后续三步共用）
            filtered_cands = cand_filter.filter_candidates(tcm_diag_name=predicted_diag)

            # Step 2：以中医诊断为条件，预测证型（RAG 模式注入同诊断相似病例）
            if args.mode == "rag" and retrieved:
                s2_prompt = build_step2_rag_prompt(info, filtered_cands, predicted_diag or "未知", retrieved)
            else:
                s2_prompt = build_step2_prompt(info, filtered_cands, predicted_diag or "未知")
            try:
                raw2 = _call_gpt4o(
                    client, s2_prompt, model=args.model,
                    system_prompt=SYSTEM_PROMPT_STEP2,
                    no_json_mode=args.no_json_mode,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except RuntimeError as e:
                print(f"\n[SKIP row {i} step2] {e}")
                raw2 = ""
            s2 = _parse_step2_response(raw2, filtered_cands)
            predicted_syndrome = s2["syndrome_type"]

            # Step 3：以中医诊断+证型为条件，预测治法（RAG 模式注入检索摘要）
            if args.mode == "rag" and retrieved:
                s3_prompt = build_step3_rag_prompt(
                    info, filtered_cands,
                    predicted_diag or "未知",
                    predicted_syndrome or "未知",
                    retrieved,
                )
            else:
                s3_prompt = build_step3_prompt(
                    info, filtered_cands,
                    predicted_diag or "未知",
                    predicted_syndrome or "未知",
                )
            try:
                raw3 = _call_gpt4o(
                    client, s3_prompt, model=args.model,
                    system_prompt=SYSTEM_PROMPT_STEP3,
                    no_json_mode=args.no_json_mode,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except RuntimeError as e:
                print(f"\n[SKIP row {i} step3] {e}")
                raw3 = ""
            s3 = _parse_step3_response(raw3, filtered_cands)
            predicted_treatment = s3["treatment_method"]

            # Step 4：以中医诊断+证型+治法为条件，预测草药（RAG 模式注入检索摘要）
            if args.mode == "rag" and retrieved:
                s4_prompt = build_step4_rag_prompt(
                    info, filtered_cands,
                    predicted_diag or "未知",
                    predicted_syndrome or "未知",
                    predicted_treatment,
                    retrieved,
                )
            else:
                s4_prompt = build_step4_prompt(
                    info, filtered_cands,
                    predicted_diag or "未知",
                    predicted_syndrome or "未知",
                    predicted_treatment,
                )
            try:
                raw4 = _call_gpt4o(
                    client, s4_prompt, model=args.model,
                    system_prompt=SYSTEM_PROMPT_STEP4,
                    no_json_mode=args.no_json_mode,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except RuntimeError as e:
                print(f"\n[SKIP row {i} step4] {e}")
                raw4 = ""
            s4 = _parse_step4_response(raw4, filtered_cands)

            return {
                "row_idx": i,
                "mode": f"{args.mode}_four_stage",
                "gt": {
                    "tcm_diag_name": info["gt_tcm_diag_name"],
                    "syndrome_name": info["gt_syndrome_name"],
                    "treatment_names": info["gt_treatment_names"],
                    "herb_names": info["gt_herb_names"],
                },
                "pred": {
                    "reasoning": (
                        f"Step1: {s1.get('reasoning','')} | "
                        f"Step2: {s2.get('reasoning','')} | "
                        f"Step3: {s3.get('reasoning','')} | "
                        f"Step4: {s4.get('reasoning','')}"
                    ),
                    "tcm_diagnosis": s1["tcm_diagnosis"],
                    "syndrome_type": s2["syndrome_type"],
                    "treatment_method": s3["treatment_method"],
                    "herb_recommendation": s4["herb_recommendation"],
                    "tcm_diagnosis_raw": s1["tcm_diagnosis_raw"],
                    "syndrome_type_raw": s2["syndrome_type_raw"],
                    "treatment_method_raw": s3["treatment_method_raw"],
                    "herb_recommendation_raw": s4["herb_recommendation_raw"],
                },
                "raw_response": {
                    "step1": raw1,
                    "step2": raw2,
                    "step3": raw3,
                    "step4": raw4,
                },
            }

        # ── 两阶段推理路径 ──────────────────────────────────────────
        if args.two_stage:
            # Stage 1：仅预测中医诊断
            s1_prompt = build_stage1_prompt(info, candidates)
            try:
                raw1 = _call_gpt4o(
                    client, s1_prompt, model=args.model,
                    system_prompt=SYSTEM_PROMPT_STAGE1,
                    no_json_mode=args.no_json_mode,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except RuntimeError as e:
                print(f"\n[SKIP row {i} stage1] {e}")
                raw1 = ""
            s1 = _parse_stage1_response(raw1, candidates)
            predicted_diag = s1["tcm_diagnosis"]

            # 用预测的诊断精筛候选（cand_filter 一定存在）
            filtered_cands = cand_filter.filter_candidates(tcm_diag_name=predicted_diag)

            # Stage 2：基于诊断预测证型/治法/草药
            if args.mode == "zero_shot":
                s2_prompt = build_stage2_prompt(info, filtered_cands, predicted_diag or "未知")
            elif args.mode == "few_shot":
                s2_prompt = build_stage2_prompt(
                    info, filtered_cands, predicted_diag or "未知", examples=few_shot_examples
                )
            else:  # rag
                retrieved = retriever.retrieve(info, k=args.rag_k)
                s2_prompt = build_stage2_rag_prompt(
                    info, filtered_cands, predicted_diag or "未知", retrieved
                )

            try:
                raw2 = _call_gpt4o(
                    client, s2_prompt, model=args.model,
                    system_prompt=SYSTEM_PROMPT_STAGE2,
                    no_json_mode=args.no_json_mode,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except RuntimeError as e:
                print(f"\n[SKIP row {i} stage2] {e}")
                raw2 = ""
            s2 = _parse_stage2_response(raw2, candidates)

            parsed = {
                "reasoning": s2.get("reasoning", ""),
                "tcm_diagnosis": s1["tcm_diagnosis"],
                "syndrome_type": s2.get("syndrome_type"),
                "treatment_method": s2.get("treatment_method", []),
                "herb_recommendation": s2.get("herb_recommendation", []),
                "tcm_diagnosis_raw": s1["tcm_diagnosis_raw"],
                "syndrome_type_raw": s2.get("syndrome_type_raw"),
                "treatment_method_raw": s2.get("treatment_method_raw", []),
                "herb_recommendation_raw": s2.get("herb_recommendation_raw", []),
            }
            return {
                "row_idx": i,
                "mode": f"{args.mode}_two_stage",
                "gt": {
                    "tcm_diag_name": info["gt_tcm_diag_name"],
                    "syndrome_name": info["gt_syndrome_name"],
                    "treatment_names": info["gt_treatment_names"],
                    "herb_names": info["gt_herb_names"],
                },
                "pred": parsed,
                "raw_response": {"stage1": raw1, "stage2": raw2},
            }

        # ── 单阶段推理路径（原有逻辑）────────────────────────────────
        if args.mode == "zero_shot":
            filtered_cands = cand_filter.filter_candidates(tcm_diag_name=None) if cand_filter else candidates
            prompt = build_zero_shot_prompt(info, filtered_cands)
        elif args.mode == "few_shot":
            filtered_cands = cand_filter.filter_candidates(tcm_diag_name=None) if cand_filter else candidates
            prompt = build_few_shot_prompt(info, filtered_cands, few_shot_examples)
        else:  # rag
            retrieved = retriever.retrieve(info, k=args.rag_k)
            if cand_filter is not None:
                # 用检索到的相似样本做多数投票，推测当前患者最可能的中医诊断，
                # 再用该诊断做 per-diagnosis 精筛（比全局 Top-K 更精准）
                diag_votes = _collections.Counter(
                    ex["gt_tcm_diag_name"] for ex in retrieved if ex.get("gt_tcm_diag_name")
                )
                guessed_diag = diag_votes.most_common(1)[0][0] if diag_votes else None
                filtered_cands = cand_filter.filter_candidates(tcm_diag_name=guessed_diag)
            else:
                filtered_cands = candidates
            prompt = build_rag_prompt(info, filtered_cands, retrieved)

        try:
            raw = _call_gpt4o(
                client, prompt, model=args.model,
                no_json_mode=args.no_json_mode,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except RuntimeError as e:
            print(f"\n[SKIP row {i}] {e}")
            raw = ""

        parsed = _parse_response(raw, candidates)
        return {
            "row_idx": i,
            "mode": args.mode,
            "gt": {
                "tcm_diag_name": info["gt_tcm_diag_name"],
                "syndrome_name": info["gt_syndrome_name"],
                "treatment_names": info["gt_treatment_names"],
                "herb_names": info["gt_herb_names"],
            },
            "pred": parsed,
            "raw_response": raw,
        }

    workers = max(1, args.workers)
    print(f"并发线程数: {workers}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, i): i for i in indices}
        with tqdm(total=len(indices), desc=f"GPT-4o [{args.mode}]") as pbar:
            for future in as_completed(futures):
                try:
                    record = future.result()
                    _append_result(output_path, record)
                except Exception as e:
                    idx = futures[future]
                    print(f"\n[ERROR row {idx}] {e}")
                pbar.update(1)

    print(f"\n完成！预测结果已保存到 {output_path}")


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPT-4o TCM 诊疗链推理基线")
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot", "rag"],
        default="zero_shot",
        help="推理模式：zero_shot / few_shot / rag（默认 zero_shot）",
    )
    parser.add_argument(
        "--rag_k",
        type=int,
        default=3,
        help="RAG 模式：每条测试样本检索多少个相似训练样本作为 few-shot（默认 5）",
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        help=(
            "启用两阶段推理：第一次调用仅预测中医诊断，"
            "再用诊断精筛候选集后第二次调用预测证型/治法/草药。"
            "自动启用 CandidateFilter，可与 --filter_syn_k 等参数联用。"
        ),
    )
    parser.add_argument(
        "--four_stage",
        action="store_true",
        help=(
            "启用四步严格链式推理：每步仅预测一个字段，后一步以前一步的输出为条件。"
            "Step1→中医诊断，Step2→证型，Step3→治法，Step4→草药。"
            "每个样本共 4 次 API 调用，自动启用 CandidateFilter。"
        ),
    )
    parser.add_argument(
        "--filter_candidates",
        action="store_true",
        help="启用候选标签预筛选：只保留训练集中最常见的 Top-K 证型/草药/治法",
    )
    parser.add_argument(
        "--filter_syn_k",
        type=int,
        default=30,
        help="候选过滤：保留Top-K证型（默认30，全量266）",
    )
    parser.add_argument(
        "--filter_treat_k",
        type=int,
        default=20,
        help="候选过滤：保留Top-K治法（默认20，全量160）",
    )
    parser.add_argument(
        "--filter_herb_k",
        type=int,
        default=80,
        help="候选过滤：保留Top-K草药（默认80，全量526）",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="模型名称（默认 gpt-4o）",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="输出文件标识（默认自动用模型名），多模型并跑时用于区分结果文件",
    )
    parser.add_argument(
        "--no_json_mode",
        action="store_true",
        help="禁用 response_format=json_object（部分推理模型如 deepseek-r1 不支持时使用）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度（默认 0.0；deepseek-r1 建议 0.6）",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="最大输出 token 数（默认 2048；推理模型建议 8192）",
    )
    parser.add_argument(
        "--dry_run",
        type=int,
        default=0,
        help="仅处理前 N 条（0=全量）。调试时建议先跑 --dry_run 100",
    )
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=3,
        help="Few-shot 示例数量（默认 3）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="并发线程数（默认 5），建议根据 API 配额调整，最大不超过 20",
    )
    parser.add_argument(
        "--sleep_every",
        type=int,
        default=0,
        help="（已废弃，保留兼容）每处理 N 条后暂停（0=不暂停）",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=2.0,
        help="（已废弃，保留兼容）暂停秒数（默认 2.0）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    # 自动加载 .env 文件（如果存在）
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    main()
