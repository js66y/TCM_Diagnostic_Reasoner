"""
gpt4o_baseline/prompts.py
提示词模板及格式化函数
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "你是一位经验丰富的中医临床专家，擅长辨证论治。"
    "你的任务是根据患者病历，按照【诊断→证型→治法→用药】的链式逻辑完成推理。"
    "【强制规则】"
    "① 所有输出标签必须严格来自给定候选列表，一字不差，不得自行创造或改写。"
    "② herb_recommendation 根据病情需要选取适量中药，所选中药须体现君臣佐使配伍逻辑，与治法高度对应。"
    "③ treatment_method 【绝大多数情况只需 2 个】，病情确实复杂才选 3 个，绝对不超过 3 个；所选中药必须与治法高度对应，体现君臣佐使配伍逻辑。"
    "④ 必须以合法的 JSON 格式输出，不要输出任何 JSON 以外的文字。"
)

# 两阶段推理 —— 第一阶段：仅预测中医诊断
SYSTEM_PROMPT_STAGE1 = (
    "你是一位经验丰富的中医临床专家。"
    "你的任务是根据患者病历，从给定候选列表中选出最符合的中医病名（单选）。"
    "【强制规则】"
    "① 必须从候选列表中选择，一字不差，不得自行创造或改写。"
    "② 只输出 JSON，不要有任何额外文字。"
)

# 两阶段推理 —— 第二阶段：已知诊断，预测证型/治法/草药
SYSTEM_PROMPT_STAGE2 = (
    "你是一位经验丰富的中医临床专家，擅长辨证论治。"
    "中医诊断已确定，你的任务是完成【证型→治法→用药】的链式推理。"
    "【强制规则】"
    "① 所有输出标签必须严格来自给定候选列表，一字不差，不得自行创造或改写。"
    "② herb_recommendation 根据病情需要选取适量中药，所选中药须体现君臣佐使配伍逻辑，与治法高度对应。"
    "③ treatment_method 【绝大多数情况只需 2 个】，病情确实复杂才选 3 个，绝对不超过 3 个；所选中药必须与治法高度对应，体现君臣佐使配伍逻辑。"
    "④ 必须以合法的 JSON 格式输出，不要输出任何 JSON 以外的文字。"
)

# 四步链式推理 —— Step 2：已知中医诊断，预测证型
SYSTEM_PROMPT_STEP2 = (
    "你是一位经验丰富的中医临床专家。"
    "中医诊断已确定，你的任务是根据患者病历和已知诊断，从候选列表中选出最符合的证型（单选）。"
    "【强制规则】"
    "① 必须从候选列表中选择，一字不差，不得自行创造或改写。"
    "② 只输出 JSON，不要有任何额外文字。"
)

# 四步链式推理 —— Step 3：已知中医诊断+证型，预测治法
SYSTEM_PROMPT_STEP3 = (
    "你是一位经验丰富的中医临床专家。"
    "中医诊断和证型已确定，你的任务是根据患者病历和已知诊断/证型，从候选列表中选出适用的治则治法。"
    "【强制规则】"
    "① 必须从候选列表中选择，一字不差，不得自行创造或改写。"
    "② treatment_method 【绝大多数情况只需 2 个】，病情确实复杂才选 3 个，绝对不超过 3 个。"
    "③ 只输出 JSON，不要有任何额外文字。"
)

# 四步链式推理 —— Step 4：已知中医诊断+证型+治法，预测草药
SYSTEM_PROMPT_STEP4 = (
    "你是一位经验丰富的中医临床专家，擅长遣方用药。"
    "中医诊断、证型和治法已全部确定，你的任务是根据患者病历和已知诊断/证型/治法，从候选中药列表中选出组成处方的中药。"
    "【强制规则】"
    "① 所选中药须严格来自给定候选列表，一字不差，不得自行创造或改写。"
    "② 根据病情需要选取适量中药，须体现君臣佐使配伍逻辑，与治法高度对应。"
    "③ 只输出 JSON，不要有任何额外文字。"
)

# ────────────────────────────────────────────────────────────────
# 核心 User Prompt 模板（zero-shot 直接用，few-shot 作为最后一条）
# ────────────────────────────────────────────────────────────────
_QUERY_TEMPLATE = """\
请根据以下患者病历，严格按照中医辨证论治的链式逻辑完成四步推理：

【推理步骤（必须按顺序进行，后一步依赖前一步结论）】
步骤1 · 中医诊断：从候选列表中选出最符合的中医病名（单选）
步骤2 · 证型辨析：基于步骤1的诊断，从候选列表中选出最符合的证型（单选）
步骤3 · 治法确定：基于步骤2的证型，从候选列表中选出适用的治则治法（【通常 2 个】，特殊情况最多 3 个，绝对不超过 3 个）
步骤4 · 组方用药：基于步骤3的治法，从候选中药列表中选出组成处方的中药
  - 根据病情需要选取适量中药，须体现君臣佐使配伍，与步骤3治法高度对应
  - 只能使用候选列表中原文，不得添加剂量、炮制方法或任何修饰

【输出格式】（仅输出以下 JSON，不要有任何其他文字）
{{
  "reasoning": "步骤1：…→步骤2：…→步骤3：…→步骤4：…（写出每步推理依据）",
  "tcm_diagnosis": "（单个候选标签）",
  "syndrome_type": "（单个候选标签）",
  "treatment_method": ["治法标签1", "治法标签2"],  // 通常2个，最多3个，绝对不超过3个
  "herb_recommendation": ["药名1", "药名2", ..., "药名N"]
}}

【患者基本信息】
性别：{gender}
年龄：{age}岁

【主诉】
{chief_complaint}

【现病史】
{medical_history}

【体格检查】
{physical_examination}

【西医初步诊断】
{preliminary_western_diagnosis}

---

【候选标签】（必须从以下标签中选择，不得使用标签外的内容）

中医诊断候选（单选）：
{tcm_diagnosis_candidates}

证型候选（单选）：
{syndrome_candidates}

治法候选（多选）：
{treatment_candidates}

中药候选（多选）：
{herb_candidates}

---

请按照要求完成辨证论治推理，并严格按照JSON格式输出结果。"""

# ────────────────────────────────────────────────────────────────
# Few-shot 示例模板
# ────────────────────────────────────────────────────────────────
_EXAMPLE_TEMPLATE = """\
【示例患者】
性别：{gender}
年龄：{age}岁
主诉：{chief_complaint}
现病史：{medical_history}
体格检查：{physical_examination}
西医初步诊断：{preliminary_western_diagnosis}

【示例输出】（herb_recommendation 共 {herb_count} 味，在合适范围内）
{{
  "reasoning": "步骤1：根据病历辨为{gt_tcm_diag_name}→步骤2：证属{gt_syndrome_name}→步骤3：确定治法→步骤4：据治法选药组方",
  "tcm_diagnosis": "{gt_tcm_diag_name}",
  "syndrome_type": "{gt_syndrome_name}",
  "treatment_method": {treatment_json},
  "herb_recommendation": {herb_json}
}}"""


# ────────────────────────────────────────────────────────────────
# 候选列表格式化
# ────────────────────────────────────────────────────────────────
def _format_candidates(names: list[str], sep: str = "、") -> str:
    return sep.join(names)


def _format_candidates_numbered(names: list[str], per_line: int = 10) -> str:
    """分行编号展示候选列表，让模型更容易定位和精确输出候选名称"""
    lines = []
    for i, name in enumerate(names):
        if i % per_line == 0 and i > 0:
            lines.append("\n")
        lines.append(f"{name}")
    return "、".join(names)


# ────────────────────────────────────────────────────────────────
# 公开接口
# ────────────────────────────────────────────────────────────────
def build_zero_shot_prompt(info: dict, candidates: dict) -> str:
    """构造 zero-shot user prompt"""
    return _QUERY_TEMPLATE.format(
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        tcm_diagnosis_candidates=_format_candidates(candidates["tcm_diag"]["names"]),
        syndrome_candidates=_format_candidates(candidates["syndrome"]["names"]),
        treatment_candidates=_format_candidates(candidates["treatment"]["names"]),
        herb_candidates=_format_candidates(candidates["herb"]["names"]),
    )


def build_few_shot_prompt(info: dict, candidates: dict, examples: list[dict]) -> str:
    """
    构造 few-shot user prompt：
    在正式问题前插入 k 条示例，用分隔线隔开。
    """
    import json as _json

    example_blocks = []
    for ex in examples:
        herb_names = ex["gt_herb_names"]
        block = _EXAMPLE_TEMPLATE.format(
            gender=ex["gender"],
            age=ex["age"],
            chief_complaint=ex["chief_complaint"] or "无",
            medical_history=ex["medical_history"] or "无",
            physical_examination=ex["physical_examination"] or "无",
            preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
            gt_tcm_diag_name=ex["gt_tcm_diag_name"],
            gt_syndrome_name=ex["gt_syndrome_name"],
            treatment_json=_json.dumps(ex["gt_treatment_names"], ensure_ascii=False),
            herb_json=_json.dumps(herb_names, ensure_ascii=False),
            herb_count=len(herb_names),
        )
        example_blocks.append(block)

    examples_prefix = (
        "以下是几个辨证论治推理的示例，请参考示例的推理模式：\n\n"
        + "\n\n---\n\n".join(example_blocks)
        + "\n\n---\n\n现在请对以下患者进行推理：\n\n"
    )

    main_query = _QUERY_TEMPLATE.format(
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        tcm_diagnosis_candidates=_format_candidates(candidates["tcm_diag"]["names"]),
        syndrome_candidates=_format_candidates(candidates["syndrome"]["names"]),
        treatment_candidates=_format_candidates(candidates["treatment"]["names"]),
        herb_candidates=_format_candidates(candidates["herb"]["names"]),
    )

    return examples_prefix + main_query


def _build_retrieval_summary(retrieved_examples: list[dict]) -> str:
    """
    从检索到的相似病例中提炼诊断/证型统计摘要，
    在主 query 前作为强提示，帮助模型锁定诊断范围。
    """
    import collections
    diag_counter: collections.Counter = collections.Counter()
    syn_counter: collections.Counter = collections.Counter()
    herb_counter: collections.Counter = collections.Counter()

    for ex in retrieved_examples:
        if ex.get("gt_tcm_diag_name"):
            diag_counter[ex["gt_tcm_diag_name"]] += 1
        if ex.get("gt_syndrome_name"):
            syn_counter[ex["gt_syndrome_name"]] += 1
        for h in ex.get("gt_herb_names", []):
            herb_counter[h] += 1

    lines = []

    # 诊断：按频次列出，出现多次的加★强调
    diag_parts = []
    for name, cnt in diag_counter.most_common():
        star = "★" * cnt if cnt > 1 else name
        diag_parts.append(f"{name}（出现{cnt}次）" if cnt > 1 else name)
    lines.append(f"相似病例的中医诊断（从高到低）：{'、'.join(diag_parts)}")
    lines.append("→ 步骤1选择中医诊断时，请优先考虑上述高频诊断是否符合当前患者症状。")

    # 证型
    syn_parts = [f"{n}（{c}次）" if c > 1 else n for n, c in syn_counter.most_common()]
    lines.append(f"相似病例的证型（从高到低）：{'、'.join(syn_parts)}")
    lines.append("→ 步骤2选择证型时，请优先考虑上述证型是否与你在步骤1的诊断匹配。")

    # 高频草药（出现≥2次的）
    common_herbs = [n for n, c in herb_counter.most_common() if c >= 2]
    if common_herbs:
        lines.append(f"相似病例中高频出现的草药（≥2次）：{'、'.join(common_herbs[:20])}")
        lines.append("→ 步骤4组方时，上述草药可作为核心候选，结合治法和配伍原则增减。")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# 两阶段推理 Prompt（Stage 1：仅中医诊断）
# ────────────────────────────────────────────────────────────────
_STAGE1_TEMPLATE = """\
请根据以下患者病历，从候选列表中选出最符合的中医病名（单选）。

【患者基本信息】
性别：{gender}
年龄：{age}岁

【主诉】
{chief_complaint}

【现病史】
{medical_history}

【体格检查】
{physical_examination}

【西医初步诊断】
{preliminary_western_diagnosis}

---

【中医诊断候选（单选）】
{tcm_diagnosis_candidates}

---

【输出格式】（仅输出以下 JSON，不要有任何其他文字）
{{
  "reasoning": "根据症状…，最符合的中医病名是…（简短说明依据）",
  "tcm_diagnosis": "（单个候选标签）"
}}"""


def build_stage1_prompt(info: dict, candidates: dict) -> str:
    """两阶段第一阶段：仅预测中医诊断"""
    return _STAGE1_TEMPLATE.format(
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        tcm_diagnosis_candidates=_format_candidates(candidates["tcm_diag"]["names"]),
    )


_STEP1_EXAMPLE_TEMPLATE = """\
【参考病例】
性别：{gender}
年龄：{age}岁
主诉：{chief_complaint}
现病史：{medical_history}
体格检查：{physical_examination}
西医初步诊断：{preliminary_western_diagnosis}
→ 中医诊断：{gt_tcm_diag_name}"""


def build_step1_rag_prompt(
    info: dict, candidates: dict, retrieved_examples: list[dict]
) -> str:
    """四步链式 Step1 RAG 版：展示完整相似病例（含病历+诊断）+ 仅预测中医诊断"""
    import collections as _col

    # 展示完整病例（病历原文 + 中医诊断答案），与单阶段 RAG 信息量一致
    example_blocks = []
    for ex in retrieved_examples:
        if not ex.get("gt_tcm_diag_name"):
            continue
        block = _STEP1_EXAMPLE_TEMPLATE.format(
            gender=ex["gender"],
            age=ex["age"],
            chief_complaint=ex["chief_complaint"] or "无",
            medical_history=ex["medical_history"] or "无",
            physical_examination=ex["physical_examination"] or "无",
            preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
            gt_tcm_diag_name=ex["gt_tcm_diag_name"],
        )
        example_blocks.append(block)

    # 诊断频次摘要
    diag_counter: _col.Counter = _col.Counter()
    for ex in retrieved_examples:
        if ex.get("gt_tcm_diag_name"):
            diag_counter[ex["gt_tcm_diag_name"]] += 1
    diag_parts = [
        f"{n}（{c}次）" if c > 1 else n
        for n, c in diag_counter.most_common()
    ]

    prefix = ""
    if example_blocks:
        prefix = (
            "以下是从病历库中检索到的与当前患者症状最相似的真实病例及其中医诊断，请参考其辨证规律：\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + "\n\n---\n\n"
            + f"【检索摘要】相似病例诊断分布（从高到低）：{'、'.join(diag_parts)}\n"
            + "→ 请优先考虑上述高频诊断是否符合当前患者症状，再从候选列表中选出最终答案。\n\n"
        )

    return prefix + _STAGE1_TEMPLATE.format(
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        tcm_diagnosis_candidates=_format_candidates(candidates["tcm_diag"]["names"]),
    )


# ────────────────────────────────────────────────────────────────
# 两阶段推理 Prompt（Stage 2：已知诊断，预测证型/治法/草药）
# ────────────────────────────────────────────────────────────────
_STAGE2_TEMPLATE = """\
患者的中医诊断已确定为【{tcm_diagnosis}】，请在此基础上完成辨证论治推理：

【推理步骤（从步骤2开始）】
步骤2 · 证型辨析：基于诊断【{tcm_diagnosis}】，从候选列表中选出最符合的证型（单选）
步骤3 · 治法确定：基于步骤2的证型，从候选列表中选出适用的治则治法（【通常 2 个】，特殊情况最多 3 个，绝对不超过 3 个）
步骤4 · 组方用药：基于步骤3的治法，从候选中药列表中选出组成处方的中药
  - 根据病情需要选取适量中药，须体现君臣佐使配伍，与步骤3治法高度对应
  - 只能使用候选列表中原文，不得添加剂量、炮制方法或任何修饰

【输出格式】（仅输出以下 JSON，不要有任何其他文字）
{{
  "reasoning": "步骤2：…→步骤3：…→步骤4：…（写出每步推理依据）",
  "syndrome_type": "（单个候选标签）",
  "treatment_method": ["治法标签1", "治法标签2"],
  "herb_recommendation": ["药名1", "药名2", ..., "药名N"]
}}

【患者基本信息】
性别：{gender}
年龄：{age}岁

【主诉】
{chief_complaint}

【现病史】
{medical_history}

【体格检查】
{physical_examination}

【西医初步诊断】
{preliminary_western_diagnosis}

---

【候选标签】（已根据诊断【{tcm_diagnosis}】精筛，范围更精准，必须严格从以下标签中选择）

证型候选（单选）：
{syndrome_candidates}

治法候选（多选）：
{treatment_candidates}

中药候选（多选）：
{herb_candidates}

---

请严格按照 JSON 格式输出结果。"""


def build_stage2_prompt(
    info: dict,
    candidates: dict,
    tcm_diag_name: str,
    examples: list[dict] | None = None,
) -> str:
    """
    两阶段第二阶段：已知中医诊断，预测证型/治法/草药。
    可选传入 few-shot 示例（examples）。
    """
    import json as _json

    prefix = ""
    if examples:
        example_blocks = []
        for ex in examples:
            herb_names = ex["gt_herb_names"]
            block = _EXAMPLE_TEMPLATE.format(
                gender=ex["gender"],
                age=ex["age"],
                chief_complaint=ex["chief_complaint"] or "无",
                medical_history=ex["medical_history"] or "无",
                physical_examination=ex["physical_examination"] or "无",
                preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
                gt_tcm_diag_name=ex["gt_tcm_diag_name"],
                gt_syndrome_name=ex["gt_syndrome_name"],
                treatment_json=_json.dumps(ex["gt_treatment_names"], ensure_ascii=False),
                herb_json=_json.dumps(herb_names, ensure_ascii=False),
                herb_count=len(herb_names),
            )
            example_blocks.append(block)
        prefix = (
            "以下是几个辨证论治推理的示例，请参考示例的推理模式：\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + "\n\n---\n\n现在请对以下患者进行推理：\n\n"
        )

    main_query = _STAGE2_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        syndrome_candidates=_format_candidates(candidates["syndrome"]["names"]),
        treatment_candidates=_format_candidates(candidates["treatment"]["names"]),
        herb_candidates=_format_candidates(candidates["herb"]["names"]),
    )

    return prefix + main_query


def build_stage2_rag_prompt(
    info: dict,
    candidates: dict,
    tcm_diag_name: str,
    retrieved_examples: list[dict],
) -> str:
    """两阶段第二阶段（RAG 变体）：检索相似病例 + 已知诊断，预测证型/治法/草药"""
    import json as _json

    example_blocks = []
    for ex in retrieved_examples:
        herb_names = ex["gt_herb_names"]
        if not herb_names:
            continue
        block = _EXAMPLE_TEMPLATE.format(
            gender=ex["gender"],
            age=ex["age"],
            chief_complaint=ex["chief_complaint"] or "无",
            medical_history=ex["medical_history"] or "无",
            physical_examination=ex["physical_examination"] or "无",
            preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
            gt_tcm_diag_name=ex["gt_tcm_diag_name"],
            gt_syndrome_name=ex["gt_syndrome_name"],
            treatment_json=_json.dumps(ex["gt_treatment_names"], ensure_ascii=False),
            herb_json=_json.dumps(herb_names, ensure_ascii=False),
            herb_count=len(herb_names),
        )
        example_blocks.append(block)

    if example_blocks:
        diag_summary = _build_retrieval_summary(retrieved_examples)
        prefix = (
            "以下是从病历库中检索到的与当前患者症状最相似的真实病例，请参考其辨证用药规律：\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + f"\n\n---\n\n【检索摘要·强参考】\n{diag_summary}\n"
            + "\n现在请对以下患者进行推理（候选标签已根据诊断类型预筛选，范围更精准）：\n\n"
        )
    else:
        prefix = ""

    main_query = _STAGE2_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        syndrome_candidates=_format_candidates(candidates["syndrome"]["names"]),
        treatment_candidates=_format_candidates(candidates["treatment"]["names"]),
        herb_candidates=_format_candidates(candidates["herb"]["names"]),
    )

    return prefix + main_query


# ────────────────────────────────────────────────────────────────
# 四步链式推理 Prompt（Step 2：已知诊断，仅预测证型）
# ────────────────────────────────────────────────────────────────
_STEP2_TEMPLATE = """\
中医诊断已确定为【{tcm_diagnosis}】，请根据以下患者病历，从候选列表中选出最符合的证型（单选）。

【已确定结论】
中医诊断：{tcm_diagnosis}

【患者基本信息】
性别：{gender}
年龄：{age}岁

【主诉】
{chief_complaint}

【现病史】
{medical_history}

【体格检查】
{physical_examination}

【西医初步诊断】
{preliminary_western_diagnosis}

---

【证型候选（单选）】（已根据诊断【{tcm_diagnosis}】精筛）
{syndrome_candidates}

---

【输出格式】（仅输出以下 JSON，不要有任何其他文字）
{{
  "reasoning": "基于诊断【{tcm_diagnosis}】及患者症状，证属…（简短说明依据）",
  "syndrome_type": "（单个候选标签）"
}}"""


def build_step2_prompt(info: dict, candidates: dict, tcm_diag_name: str) -> str:
    """四步链式 Step2：已知中医诊断，预测证型"""
    return _STEP2_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        syndrome_candidates=_format_candidates(candidates["syndrome"]["names"]),
    )


_STEP2_EXAMPLE_TEMPLATE = """\
【参考病例】
性别：{gender}
年龄：{age}岁
主诉：{chief_complaint}
现病史：{medical_history}
体格检查：{physical_examination}
西医初步诊断：{preliminary_western_diagnosis}
→ 中医诊断：{gt_tcm_diag_name}
→ 证型：{gt_syndrome_name}"""


def build_step2_rag_prompt(
    info: dict,
    candidates: dict,
    tcm_diag_name: str,
    retrieved_examples: list[dict],
) -> str:
    """四步链式 Step2 RAG 版：参考病例仅展示「病历+诊断+证型」，不含治法/草药"""
    # 优先展示与预测诊断相同的病例
    same_diag = [ex for ex in retrieved_examples if ex.get("gt_tcm_diag_name") == tcm_diag_name]
    other = [ex for ex in retrieved_examples if ex.get("gt_tcm_diag_name") != tcm_diag_name]
    ref_examples = (same_diag + other)[:len(retrieved_examples)]

    example_blocks = []
    for ex in ref_examples:
        if not ex.get("gt_syndrome_name"):
            continue
        block = _STEP2_EXAMPLE_TEMPLATE.format(
            gender=ex["gender"],
            age=ex["age"],
            chief_complaint=ex["chief_complaint"] or "无",
            medical_history=ex["medical_history"] or "无",
            physical_examination=ex["physical_examination"] or "无",
            preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
            gt_tcm_diag_name=ex["gt_tcm_diag_name"],
            gt_syndrome_name=ex["gt_syndrome_name"],
        )
        example_blocks.append(block)

    prefix = ""
    if example_blocks:
        same_note = f"（其中 {len(same_diag)} 例与当前诊断【{tcm_diag_name}】相同）" if same_diag else ""
        prefix = (
            f"以下是与当前患者症状最相似的真实病例{same_note}，"
            "参考病例仅展示【中医诊断+证型】，请据此推断当前患者证型：\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + "\n\n---\n\n"
        )

    return prefix + _STEP2_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        syndrome_candidates=_format_candidates(candidates["syndrome"]["names"]),
    )


_STEP3_EXAMPLE_TEMPLATE = """\
【参考病例】
性别：{gender}
年龄：{age}岁
主诉：{chief_complaint}
现病史：{medical_history}
体格检查：{physical_examination}
西医初步诊断：{preliminary_western_diagnosis}
→ 中医诊断：{gt_tcm_diag_name}
→ 证型：{gt_syndrome_name}
→ 治法：{gt_treatment_str}"""


def build_step3_rag_prompt(
    info: dict,
    candidates: dict,
    tcm_diag_name: str,
    syndrome_name: str,
    retrieved_examples: list[dict],
) -> str:
    """四步链式 Step3 RAG 版：参考病例仅展示「病历+诊断+证型+治法」，不含草药"""
    import json as _json
    same_diag = [ex for ex in retrieved_examples if ex.get("gt_tcm_diag_name") == tcm_diag_name]
    other = [ex for ex in retrieved_examples if ex.get("gt_tcm_diag_name") != tcm_diag_name]
    ref_examples = (same_diag + other)[:len(retrieved_examples)]

    example_blocks = []
    for ex in ref_examples:
        if not ex.get("gt_treatment_names"):
            continue
        block = _STEP3_EXAMPLE_TEMPLATE.format(
            gender=ex["gender"],
            age=ex["age"],
            chief_complaint=ex["chief_complaint"] or "无",
            medical_history=ex["medical_history"] or "无",
            physical_examination=ex["physical_examination"] or "无",
            preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
            gt_tcm_diag_name=ex["gt_tcm_diag_name"],
            gt_syndrome_name=ex["gt_syndrome_name"] or "未知",
            gt_treatment_str="、".join(ex["gt_treatment_names"]),
        )
        example_blocks.append(block)

    prefix = ""
    if example_blocks:
        same_note = f"（其中 {len(same_diag)} 例与当前诊断【{tcm_diag_name}】相同）" if same_diag else ""
        prefix = (
            f"以下是与当前患者症状最相似的真实病例{same_note}，"
            "参考病例仅展示【中医诊断+证型+治法】，请据此推断当前患者治法：\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + "\n\n---\n\n"
        )

    return prefix + _STEP3_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        syndrome_type=syndrome_name,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        treatment_candidates=_format_candidates(candidates["treatment"]["names"]),
    )


_STEP4_EXAMPLE_TEMPLATE = """\
【参考病例】
性别：{gender}
年龄：{age}岁
主诉：{chief_complaint}
现病史：{medical_history}
体格检查：{physical_examination}
西医初步诊断：{preliminary_western_diagnosis}
→ 中医诊断：{gt_tcm_diag_name}
→ 证型：{gt_syndrome_name}
→ 治法：{gt_treatment_str}
→ 草药（共{herb_count}味）：{gt_herb_str}"""


def build_step4_rag_prompt(
    info: dict,
    candidates: dict,
    tcm_diag_name: str,
    syndrome_name: str,
    treatment_names: list[str],
    retrieved_examples: list[dict],
) -> str:
    """四步链式 Step4 RAG 版：参考病例展示全部四个字段（含草药），用于组方参考"""
    same_diag = [ex for ex in retrieved_examples if ex.get("gt_tcm_diag_name") == tcm_diag_name]
    other = [ex for ex in retrieved_examples if ex.get("gt_tcm_diag_name") != tcm_diag_name]
    ref_examples = (same_diag + other)[:len(retrieved_examples)]

    example_blocks = []
    for ex in ref_examples:
        herb_names = ex.get("gt_herb_names", [])
        if not herb_names:
            continue
        block = _STEP4_EXAMPLE_TEMPLATE.format(
            gender=ex["gender"],
            age=ex["age"],
            chief_complaint=ex["chief_complaint"] or "无",
            medical_history=ex["medical_history"] or "无",
            physical_examination=ex["physical_examination"] or "无",
            preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
            gt_tcm_diag_name=ex["gt_tcm_diag_name"],
            gt_syndrome_name=ex["gt_syndrome_name"] or "未知",
            gt_treatment_str="、".join(ex.get("gt_treatment_names", [])) or "未知",
            herb_count=len(herb_names),
            gt_herb_str="、".join(herb_names),
        )
        example_blocks.append(block)

    treatment_str = "、".join(treatment_names) if treatment_names else "未知"
    prefix = ""
    if example_blocks:
        same_note = f"（其中 {len(same_diag)} 例与当前诊断【{tcm_diag_name}】相同）" if same_diag else ""
        prefix = (
            f"以下是与当前患者症状最相似的真实病例{same_note}，"
            "参考病例展示全部字段【中医诊断+证型+治法+草药】，请据此组方：\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + "\n\n---\n\n"
        )

    return prefix + _STEP4_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        syndrome_type=syndrome_name,
        treatment_method_str=treatment_str,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        herb_candidates=_format_candidates(candidates["herb"]["names"]),
    )


# ────────────────────────────────────────────────────────────────
# 四步链式推理 Prompt（Step 3：已知诊断+证型，仅预测治法）
# ────────────────────────────────────────────────────────────────
_STEP3_TEMPLATE = """\
中医诊断为【{tcm_diagnosis}】，证型为【{syndrome_type}】，请根据以下患者病历，从候选列表中选出适用的治则治法。

【已确定结论】
中医诊断：{tcm_diagnosis}
证型：{syndrome_type}

【患者基本信息】
性别：{gender}
年龄：{age}岁

【主诉】
{chief_complaint}

【现病史】
{medical_history}

【体格检查】
{physical_examination}

【西医初步诊断】
{preliminary_western_diagnosis}

---

【治法候选（多选）】（已根据诊断【{tcm_diagnosis}】精筛）
{treatment_candidates}

---

【输出格式】（仅输出以下 JSON，不要有任何其他文字）
{{
  "reasoning": "基于证型【{syndrome_type}】，治法应…（简短说明依据）",
  "treatment_method": ["治法标签1", "治法标签2"]
}}
注意：treatment_method 【绝大多数情况只需 2 个】，病情确实复杂才选 3 个，绝对不超过 3 个。"""


def build_step3_prompt(
    info: dict, candidates: dict, tcm_diag_name: str, syndrome_name: str
) -> str:
    """四步链式 Step3：已知中医诊断+证型，预测治法"""
    return _STEP3_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        syndrome_type=syndrome_name,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        treatment_candidates=_format_candidates(candidates["treatment"]["names"]),
    )


# ────────────────────────────────────────────────────────────────
# 四步链式推理 Prompt（Step 4：已知诊断+证型+治法，仅预测草药）
# ────────────────────────────────────────────────────────────────
_STEP4_TEMPLATE = """\
中医诊断为【{tcm_diagnosis}】，证型为【{syndrome_type}】，治法为【{treatment_method_str}】，请根据以下患者病历，从候选中药列表中选出组成处方的中药。

【已确定结论】
中医诊断：{tcm_diagnosis}
证型：{syndrome_type}
治法：{treatment_method_str}

【患者基本信息】
性别：{gender}
年龄：{age}岁

【主诉】
{chief_complaint}

【现病史】
{medical_history}

【体格检查】
{physical_examination}

【西医初步诊断】
{preliminary_western_diagnosis}

---

【中药候选（多选）】（已根据诊断【{tcm_diagnosis}】精筛）
{herb_candidates}

---

【输出格式】（仅输出以下 JSON，不要有任何其他文字）
{{
  "reasoning": "基于治法【{treatment_method_str}】，选取以下中药组方…（简短说明配伍思路）",
  "herb_recommendation": ["药名1", "药名2", "..."]
}}
注意：所选中药须体现君臣佐使配伍逻辑，与治法【{treatment_method_str}】高度对应，只能使用候选列表中的原文。"""


def build_step4_prompt(
    info: dict,
    candidates: dict,
    tcm_diag_name: str,
    syndrome_name: str,
    treatment_names: list[str],
) -> str:
    """四步链式 Step4：已知中医诊断+证型+治法，预测草药"""
    treatment_str = "、".join(treatment_names) if treatment_names else "未知"
    return _STEP4_TEMPLATE.format(
        tcm_diagnosis=tcm_diag_name,
        syndrome_type=syndrome_name,
        treatment_method_str=treatment_str,
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        herb_candidates=_format_candidates(candidates["herb"]["names"]),
    )


def build_rag_prompt(
    info: dict,
    candidates: dict,
    retrieved_examples: list[dict],
) -> str:
    """
    RAG few-shot prompt：
    - retrieved_examples 由 CaseRetriever.retrieve() 返回，是与当前患者最相似的训练样本
    - candidates 可以是经过 CandidateFilter 过滤后的精简候选集
    """
    import json as _json

    example_blocks = []
    for ex in retrieved_examples:
        herb_names = ex["gt_herb_names"]
        if not herb_names:
            continue
        block = _EXAMPLE_TEMPLATE.format(
            gender=ex["gender"],
            age=ex["age"],
            chief_complaint=ex["chief_complaint"] or "无",
            medical_history=ex["medical_history"] or "无",
            physical_examination=ex["physical_examination"] or "无",
            preliminary_western_diagnosis=ex["preliminary_western_diagnosis"] or "无",
            gt_tcm_diag_name=ex["gt_tcm_diag_name"],
            gt_syndrome_name=ex["gt_syndrome_name"],
            treatment_json=_json.dumps(ex["gt_treatment_names"], ensure_ascii=False),
            herb_json=_json.dumps(herb_names, ensure_ascii=False),
            herb_count=len(herb_names),
        )
        example_blocks.append(block)

    if example_blocks:
        # 从检索样本中提炼诊断/证型摘要，显式作为强提示
        diag_summary = _build_retrieval_summary(retrieved_examples)
        examples_prefix = (
            "以下是从病历库中检索到的与当前患者症状最相似的真实病例，请参考其辨证用药规律：\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + f"\n\n---\n\n【检索摘要·强参考】\n{diag_summary}\n"
            + "\n现在请对以下患者进行推理（候选标签已根据诊断类型预筛选，范围更精准）：\n\n"
        )
    else:
        examples_prefix = ""

    main_query = _QUERY_TEMPLATE.format(
        gender=info["gender"],
        age=info["age"],
        chief_complaint=info["chief_complaint"] or "无",
        medical_history=info["medical_history"] or "无",
        physical_examination=info["physical_examination"] or "无",
        preliminary_western_diagnosis=info["preliminary_western_diagnosis"] or "无",
        tcm_diagnosis_candidates=_format_candidates(candidates["tcm_diag"]["names"]),
        syndrome_candidates=_format_candidates(candidates["syndrome"]["names"]),
        treatment_candidates=_format_candidates(candidates["treatment"]["names"]),
        herb_candidates=_format_candidates(candidates["herb"]["names"]),
    )

    return examples_prefix + main_query
