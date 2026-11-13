# 差旅 Agent 评测

## 两种模式

```powershell
# 确定性指标：意图快车道 / 槽位解析 / 差标召回 / 审批链
python -m evals.run_eval --mode deterministic

# 端到端：真实模型，记录轮次 / 无效工具调用 / token / 时延
python -m evals.run_eval --mode live --router on
python -m evals.run_eval --mode live --router off   # 单 ReAct 基线
```

## 指标定义

| 指标 | 定义 |
|---|---|
| 一轮 | 编排器的一次 `stream_turn` 调用 |
| 无效工具调用 | 返回 `invalid_arguments` / `missing_arguments` / `precondition_not_met` / `unknown_tool` 的调用 |
| 任务成功 | 期望工具集合被调用，且 `finish_reason` 符合预期 |
| 时延 | 单次 `/chat` 端到端墙钟，报告 P50/P95 |

## 写简历前的硬要求

1. 同一数据集、同一模型版本、`temperature=0`；
2. `--router on` 与 `--router off` 各跑一遍，才有 A/B；
3. 报告落到 `evals/reports/`，面试时能打开；
4. **没有报告的数字不要写进简历。**

`deterministic` 模式的数字只能证明规则与政策引擎没退化，不能替代模型在环评测。
