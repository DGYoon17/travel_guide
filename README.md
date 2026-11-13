# TravelMind AI｜企业差旅助手

面向企业差旅场景的对话式 Agent 后端。员工用自然语言查询差标、规划行程、检索机票酒店并发起预订；服务端维护多轮行程槽位、执行差标校验、生成审批链，并把不可逆的预订/提交审批放进确认门。

技术栈：`FastAPI + AsyncIO + Pydantic + LLM Function Calling + RAG + Redis + PostgreSQL + Milvus + OpenTelemetry`。

> 本仓库同时保留了采购域的实现（`app/domain/procurement`、`app/agent/orchestrator.py`）作为同一套 harness 的参考域，**主线与默认注册表是差旅**。两域共享工具注册表、确认门、预算、短期记忆与审计。

---

## 1. 核心能力

- **差标问答**：舱位、住宿、提前预订天数、审批阈值，全部由确定性政策引擎计算，模型只负责转述
- **行程规划**：根据出发地/目的地/日期/舱位/住宿需求生成方案，给出总价、超标项与审批链
- **机票/酒店检索**：返回带有效期的报价快照（`offer_id` + 价格 + 退改规则）
- **预订辅助**：组合预订机票与酒店；任一航段失败会回滚已成功订单，不出现"机票成功、酒店失败但用户以为都成了"
- **写操作确认门**：预订、取消、提交审批都必须由用户在独立接口上带 `action_id` 显式确认
- **确认前二次核价**：报价变化超过 2% 直接中止，让用户按新价格重新确认
- **真流式输出**：内容分片推送，工具调用与执行结果作为独立 SSE 事件
- **规则降级**：模型不可用时按差标给出确定性答复，而不是 500 或空白

## 2. 多范式编排

不是"所有请求都走一个 ReAct 循环"，而是按意图、置信度与行程复杂度选择执行路径：

| 范式 | 触发条件 | 执行方式 |
|---|---|---|
| `DIRECT` | 差标/制度问答、闲聊 | 单工具 + 一次生成，不进入多轮循环 |
| `REACT` | 槽位不全、预订/审批等写操作、单域检索 | 原生 function calling 循环 + 确认门 |
| `PLAN_EXECUTE` | 行程要素齐全的规划请求 | 固定模板计划：查差标 → 搜航班 → 搜酒店 → 合规校验 → 出方案 |
| `CLARIFY` | 意图置信度 `< 0.85`，或规则/模型都没给出可信意图 | 不调工具，先澄清 |

路由是**确定性规则**（`app/core/agent/router.py`），可单测、可解释、可回归；LLM 只参与意图分类，不决定范式。

意图识别是快慢双车道（`app/core/intent/travel.py`）：

- 强模式命中 → 置信度 `0.86+`，直接短路；
- 只命中弱模式 → `0.68~0.80`，进入慢车道或澄清；
- 规则完全未命中 → `UNKNOWN`，交给路由澄清，而不是伪装成 `GENERAL`。

阈值两侧都可达，这是"低置信度触发校验"能成立的前提。

## 3. 差旅制度引擎

`app/domain/travel/policy.py` 是确定性规则，不经过模型：

- 舱位：职级 × 国内/国际 × 飞行时长（长航线可放宽一档）
- 住宿：城市等级（一线/新一线/其他）× 职级，按每晚上限计算超标金额
- 提前预订：国内 3 天、国际 7 天
- 硬约束：证件未就绪、目的地受限、日期倒置 → `block`
- 审批链：金额决定财务 BP / 分管高管，国际行程加合规审核，超标加费用合规复核

每条结论都带 `code`、`expected` 与可选的 `over_standard_cny`，模型只能转述，不能发明差标。

## 4. 记忆分层

| 层 | 内容 | 存储 | 关键约束 |
|---|---|---|---|
| L0 工作记忆 | 本轮 tool_calls 轨迹 | 进程内 | 请求结束即弃 |
| L1 会话记忆 | 对话消息 + 结构化滚动摘要 | Redis | 裁剪单位是消息组，摘要位于不可裁剪的 pinned 区 |
| L2 行程草稿 | `TripDraft` 槽位 + 方案快照 | Redis，带版本号 | 已确认字段禁止模型静默改写 |
| L3 主体画像 | 工号/部门/职级/成本中心 | HR/OA 主数据（只读） | 只从凭据与主数据来，模型不可见 |
| L4 长期偏好 | 航司/酒店/座位/餐食偏好 | Milvus 独立 collection | 写入判别 + PII 过滤 + TTL + 制度版本 |
| L5 事实日志 | 谁在何时确认/预订了什么 | PostgreSQL append-only | 只写不改，不被摘要压缩 |

短期记忆按"消息组"裁剪：assistant 的 `tool_calls` 与其全部 tool 结果绑定为不可拆原子组，避免孤立 `tool_call_id` 触发上游 400；摘要在 `pinned` 区，物理上不会被窗口裁掉。

摘要走结构化 JSON，金额/日期必须能在原文或 L2 草稿中溯源，否则丢弃并告警——采购与差旅都怕"15000 被压成 1.5 万，再被读成单价"。

## 5. 预订写路径

这是差旅与普通问答的核心区别：

1. `plan_trip` 生成方案，把 `plan_id`、航段、总价、审批链写入 L2；
2. `book_trip` 只登记 `PendingAction`，不执行；
3. 确认门保存每个 `offer_id` 的**报价快照**；
4. `/approvals/confirm` 执行前重新核价，价差 > 2% 返回 `409` 并中止；
5. 组合预订任一航段失败 → 回滚已成功订单；回滚也失败时在错误里明确列出；
6. 成功后才写订单号、审计、幂等结果，并把订单 upsert 到 PostgreSQL `travel_bookings`

身份字段（工号/职级/部门/租户）不在模型可见的 schema 里，执行前由服务端注入；确认门存储的参数会剔除这些字段，避免一次被篡改的 Redis 记录改变"以谁的身份下单"。

## 6. 可观测性

- OpenTelemetry 自动埋点（FastAPI/httpx）+ OTLP/Console exporter
- 手工 Span 覆盖：`chat.turn`、`intent.recognize`、`agent.route`、`memory.recall/compress`、`tool.invoke`、`llm.turn`、`confirm.execute`
- `X-Trace-Id` 贯通请求、审计与结构化日志；`/metrics` 暴露工具 P50、成功率、意图/范式分布、降级与预算触顶计数
- 审计事件带 `policy_version` / `prompt_version` / 工具参数与结果

## 7. 技术栈与选型

| 技术 | 用法 | 为什么 |
|---|---|---|
| FastAPI | 对话、确认、文档、指标 API；SSE | 异步友好、自动 OpenAPI |
| AsyncIO | LLM、检索、Redis/DB 全异步；只读工具并发 | IO 密集型，吞吐与首 token 延迟受益 |
| Pydantic | 入参出参、领域模型、function schema 派生 | 消掉手写 schema 与模型的双份真相 |
| Redis | L1 会话、L2 草稿、待确认动作、幂等键 | 读写快；TTL + 版本号 CAS |
| PostgreSQL | L5 审计事实日志（append-only）+ 订单表 `travel_bookings` | 差旅预订有财务后果，需不可变留痕与可对账订单 |
| Milvus | 制度知识库 + 长期偏好（两个 collection） | 语义检索；分开避免制度文档与用户偏好互相召回 |
| DashScope / OpenAI | qwen-max 或 GPT，OpenAI 兼容协议 | 一个 Provider 配置切换，不绑死供应商 |
| OpenTelemetry | 自动埋点 + 手工 Span | 排障要能分辨模型慢、检索慢还是工具慢 |
| pytest | 差旅/采购域测试 + 金标门禁 | 意图/槽位/差标/审批链退化必须被拦住 |

## 8. 目录

```text
app/
├─ agent/
│  ├─ travel_orchestrator.py   # 差旅主链路：路由分发 + 确认门 + 核价 + 降级
│  └─ orchestrator.py          # 采购域参考实现（legacy）
├─ core/
│  ├─ agent/router.py          # DIRECT / REACT / PLAN_EXECUTE / CLARIFY
│  ├─ intent/travel.py         # 差旅意图快慢车道
│  ├─ memory/trip_draft.py     # L2 行程草稿
│  ├─ memory/travel_session.py # 差旅会话持久化
│  ├─ tools/travel_builtin.py  # 11 个差旅工具
│  ├─ prompt/travel_state.py   # 状态机与可用工具集
│  └─ harness/                 # 预算 / 确认门 / 降级
├─ domain/travel/              # 差旅模型、政策引擎
├─ infrastructure/suppliers/   # 供应方适配层（当前 mock）
└─ infrastructure/observability/spans.py
evals/                         # 离线评测（确定性 + 模型在环）
scripts/                       # 冒烟与无 pytest 测试运行器
tests/golden/travel_dataset.py # 差旅金标门禁
```

## 9. 本地运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# 至少配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，以及 AUTH_DEV_TOKEN

docker compose up -d postgres redis etcd minio milvus-standalone
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger: `http://127.0.0.1:8000/docs`

### 测试与评测

```powershell
pytest tests -q
python -m evals.run_eval --mode deterministic
python -m evals.run_eval --mode live --router on    # 需要真实 API key
python -m evals.run_eval --mode live --router off   # 单 ReAct 基线
python scripts/smoke_travel.py                      # 无 pytest 环境的冒烟
```

## 10. API 示例

```json
POST /api/v1/chat
{
  "message": "帮我安排下周三去北京出差，住一晚，预算 4000",
  "session_id": "demo-session"
}
```

响应含 `session_id`、`trace_id`、`tool_calls`、`missing_fields`、`degraded`、`finish_reason`，以及本轮 `intent` / `paradigm`。

写操作分两步：

```json
{"finish_reason": "confirmation_required",
 "pending_action": {"action_id": "act_...", "tool_name": "book_trip", "preview": "..."}}
```

```json
POST /api/v1/approvals/confirm
{"action_id": "act_...", "decision": "confirm"}
```

重复执行同一方案（幂等键相同）返回首次结果；已完成的 `action_id` 再次确认返回 `404`；
报价变化返回 `409`。

## 11. 评测口径

`python -m evals.run_eval --mode deterministic` 当前结果：

| 指标 | 结果 |
|---|---|
| 意图快车道 | 12/12 |
| 槽位解析 | 6/6 |
| 差标召回 | 6/6 |
| 审批链 | 5/5 |

这些是**规则与政策引擎**的确定性指标，不包含模型在环质量。轮次、无效工具调用率、token 与时延必须用 `--mode live` 在固定数据集、固定模型版本、`temperature=0` 下实测；仓库不预置任何性能结论。

## 12. 已知限制

- 供应方是 `MockTravelSupplier`，通过 `TravelSupplier` 协议抽象；接 TMC/航司时替换实现，**当前没有真实出票能力**
- 长期偏好写入需要 Milvus 可用；未连接时自动跳过，不影响主链路
- 金标只覆盖确定性部分；模型在环评测需要 API key 与固定数据集
- 采购域保留为参考实现，默认注册表、主入口与文档均以差旅为准
- `app/core/harness/fallback.py`、`app/core/prompt/templates.py` 仍是采购域遗留，差旅使用自己的降级与提示词

## 13. 设计文档

| 文档 | 内容 |
|---|---|
| [docs/04-差旅Agent改造说明.md](docs/04-差旅Agent改造说明.md) | 领域映射、四范式路由、差标引擎、预订安全、评测结果、与简历的对应关系 |
| [docs/01-记忆与Harness改造纪要.md](docs/01-记忆与Harness改造纪要.md) | 采购版历史文档：L0–L5 分层、P0→P2 实施明细 |
| [docs/02-完整Agent应用评估与补充路线.md](docs/02-完整Agent应用评估与补充路线.md) | 采购版历史文档：成熟度评估与补齐路线 |
| [docs/03-工具调用机制详解.md](docs/03-工具调用机制详解.md) | 采购版历史文档：工具调用生命周期 |

## 14. 压测

完整操作见 [benchmarks/README.md](benchmarks/README.md)。分三层：

```powershell
# A. 编排层（假模型固定时延，不烧钱）
python benchmarks/bench_inproc.py --concurrency 1 10 30 --requests 15 --delay-ms 50

# B. HTTP/SSE（OpenAI 兼容 mock）
python -m uvicorn benchmarks.mock_openai_server:app --port 18081
python benchmarks/load_http.py --url http://127.0.0.1:18080 --token bench-token `
  --concurrency 1 5 10 --requests 10 --message "公司差旅住宿标准是多少"

# C. 真实模型（小并发，必须固定模型版本与 temperature）
python benchmarks/load_http.py --url http://127.0.0.1:8000 --token $env:AUTH_DEV_TOKEN `
  --concurrency 1 3 5 --requests 20 --message "公司差旅住宿标准是多少"
```

A/B 开关是 `ROUTER_MODE=on/off`：`off` 走单 ReAct 基线，用于对比"多范式路由少调几次模型"。

本机 mock 实验（每次模型调用 20ms，**不是真实模型结论**）：

| 模式 | 并发 | RPS | TTFT P50 | 总时长 P50 |
|---|---:|---:|---:|---:|
| router on | 1 | 20.8 | 31.1ms | 34.0ms |
| router off | 1 | 6.8 | 80.4ms | 83.3ms |
| router on | 10 | 89.3 | 74.4ms | 94.5ms |
| router off | 10 | 49.8 | 130.6ms | 149.2ms |

差异来自 baseline 每个请求多一次模型调用，不是"代码变快"。**简历里的性能数字只能用 C 层真实模型报告。**
