# GPT-4o 诊疗链推理基线

## 目录结构

```
gpt4o_baseline/
├── utils.py                  # 标签加载、数据集分割复现
├── prompts.py                # 提示词模板（zero-shot / few-shot）
├── run_gpt4o_baseline.py     # 主推理脚本
├── evaluate_gpt4o.py         # 评估脚本
└── results/
    ├── zero_shot_predictions.jsonl   # 自动生成
    ├── few_shot_predictions.jsonl    # 自动生成
    └── metrics_report.json           # 评估报告（--save 后生成）
```

## 环境准备

### 1. 安装依赖
```powershell
pip install openai tqdm scikit-learn pandas
```

### 2. 配置 API Key
在项目根目录新建 `.env` 文件（参考 `.env.example`）：
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 如使用代理地址：
# OPENAI_BASE_URL=https://your-proxy.com/v1
```

或直接在 PowerShell 中设置环境变量：
```powershell
$env:OPENAI_API_KEY = "sk-xxxxxxxx"
$env:OPENAI_BASE_URL = "https://your-proxy.com/v1"   # 可选
```

## 使用方法

### Step 1：先跑 100 条 dry-run，确认输出格式和费用

```powershell
# Zero-shot 100 条
python gpt4o_baseline/run_gpt4o_baseline.py --mode zero_shot --dry_run 100

# Few-shot 100 条
python gpt4o_baseline/run_gpt4o_baseline.py --mode few_shot --dry_run 100
```

### Step 2：确认 OK 后全量跑测试集（~7500 条）

```powershell
python gpt4o_baseline/run_gpt4o_baseline.py --mode zero_shot
python gpt4o_baseline/run_gpt4o_baseline.py --mode few_shot
```

> **断点续传**：中途中断后重新运行，已完成的条目会自动跳过。

### Step 3：评估

```powershell
# 单独评估
python gpt4o_baseline/evaluate_gpt4o.py --mode zero_shot
python gpt4o_baseline/evaluate_gpt4o.py --mode few_shot

# 同时评估两个模式并保存报告
python gpt4o_baseline/evaluate_gpt4o.py --mode zero_shot few_shot --save
```

## 评估指标

与神经符号模型完全一致，便于直接比较：

| 任务 | 主指标 | 说明 |
|------|--------|------|
| 中医诊断 | macro-F1 | 单选，187 类 |
| 证型 | macro-F1 | 单选，266 类 |
| 治法 | micro-F1 | 多选，160 类 |
| 草药 | micro-F1 | 多选，526 类 |
| Chain Score | 四项均值 | 总体得分 |

## 费用估算

| 版本 | API 调用 | 估算费用 |
|------|----------|----------|
| Zero-shot × 7500 | 7,500 次 | ~$170 USD |
| Few-shot × 7500 | 7,500 次 | ~$190 USD（prompt 更长）|
| **合计** | 15,000 次 | **~$360 USD** |

> 建议先用 `--dry_run 100` 测试 100 条，确认结果符合预期再全量运行。
