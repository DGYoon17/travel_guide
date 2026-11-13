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
- **实时数据（MCP）**：12306 车次、高德自驾/酒店/天气、黄历与航班按需查询；按行程日期与预算过滤，长途自动补航班，服务不可用时如实降级（见第 6 节）

## 2. 多范式编排

不是"所有请求都走一个 ReAct 循环"，而是按意图、置信度与行程复杂度选择执行路径：

| 范式 | 触发条件 | 执行方式 |
|---|---|---|
| `DIRECT` | 差标/制度问答、闲聊 | 单工具 + 一次生成，不进入多轮循环 |
| `REACT` | 槽位不全、预订/审批等写操作、单域检索 | 原生 function calling 循环 + 确认门 |
| `PLAN_EXECUTE` | 行程要素齐全的规划请求 | 固定模板计划：查差标 → 搜航班 → 搜酒店 → 合规校验 → 出方案 → 补车次/自驾/天气/黄历 |
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

## 6. 实时数据接入（MCP）

车次、路况、酒店、天气、黄历、航班这些**实时且易变**的事实不写进知识库，也不允许模型凭常识回答，而是通过 MCP（Model Context Protocol）按需查询。

| 能力 | 工具 | 关键处理 |
|---|---|---|
| 12306 火车票 | `query_train_tickets` | 先查城市站点代码（缓存 24h），失败退回城市名再查 |
| 高德自驾 | `plan_driving_route` | 地理编码 → 驾车路径；取最短备选路线的距离/耗时/过路费 |
| 高德酒店 | `search_local_hotels` | 按每晚预算生成关键词（高端/品牌/经济型），再按解析价格二次过滤 |
| 高德天气 | `query_weather_forecast` | 按行程日期逐日匹配，超出预报范围明确说明 |
| 八字黄历 | `query_lunar_almanac` | 农历、干支、宜忌；对"宜/忌出行"给结论，仅作文化参考 |
| 航班 | `query_flights` | 城市名 → IATA 三字码；城际距离 ≥800km 时自动追加 |
| 城际交通对比 | `plan_intercity_transport` | 一次调用拿到火车 + 自驾 +（长途）航班，给出确定性建议 |

设计约束：

- **只读**：全部是 `ToolAccess.READ`，不产生业务写入，也不进确认门；
- **宁可缺，不许编**：服务不可用时返回 `upstream_error` 并明确"实时数据不可用"，提示模型不要用常识补齐；
- **降级可标注**：高德不可用时自驾距离退化为"城市中心点 ×1.25"的估算，标记 `estimated=true` 并如实告知，过路费一律不给；
- **预算内完成**：PLAN_EXECUTE 中实时查询排在方案生成之后；剩余时间不足时跳过，单项超时会被中止而不是拖垮整轮；
- **按需注册**：未配置任何 MCP 服务时这 7 个工具根本不进 function schema，模型不会看到一批永远失败的工具。

传输层同时支持两种 MCP 形态：`/sse` 的 SSE 事件流（先收 `endpoint` 事件，再 POST，按 JSON-RPC id 配对）与普通 HTTP JSON-RPC。配置、排障与测试方式见 [docs/06-实时数据MCP集成说明.md](docs/06-实时数据MCP集成说明.md)。

## 7. 技能层（Skills，设计稿）

> 状态：**设计稿，运行时尚未加载**。完整方案与代码改造点见 [docs/07-Skills能力层设计与落地方案.md](docs/07-Skills能力层设计与落地方案.md)。

工具回答"能做什么动作"，技能回答"这类任务该怎么做、按什么顺序、输出什么、什么时候停"。当前只有一块 `BASE_SYSTEM_PROMPT` 与按 `ToolAccess` 过滤的工具集，没有任务级的过程知识；继续往提示词里堆规则，会带来体积膨胀、规则冲突无优先级、模型在 18 个工具里乱挑。

| 概念 | 回答的问题 | 落点 |
|---|---|---|
| Tool | 能做什么动作 | `ToolSpec`，`ToolAccess` 分 read/draft/write |
| Skill | 这类任务怎么做、输出什么 | 设计稿：`app/core/skills/` |
| RAG | 事实原文是什么 | `search_travel_knowledge` + Milvus |
| Policy Engine | 规则判定结果是什么 | `app/domain/travel/policy.py` |

两条硬约束：技能里不写死差标数字、价格与审批阈值（只能引用工具返回值）；技能不能覆盖 `BASE_SYSTEM_PROMPT` 的硬性规则——宪法优先于作业手册。

### 7.1 计划实现的技能

| 技能 | 触发意图 | 绑定工具 | 核心产出 |
|---|---|---|---|
| `policy-guidance` 差标解读 | `POLICY` / `RAG` | `get_travel_policy`、`check_travel_compliance`、`search_travel_knowledge` | 结论 + 数字 + 制度版本 + 原文出处 |
| `trip-planning` 行程规划 | `TRIP_PLANNING` | `plan_trip`、`search_flights`、`search_hotels`、`check_travel_compliance` + MCP | 方案 + 总价 + 超标项 + 审批链 + 数据来源 |
| `booking-gate` 预订与审批 | `BOOKING` / `APPROVAL` | `book_trip`、`submit_travel_approval`、`build_travel_request` | 待确认摘要 + 价差/退改风险 |
| `trip-servicing` 改签退订 | `TRIP_STATUS` | `get_booking_status`、`cancel_booking` | 订单快照 + 退改费 + 确认门 |
| `travel-advisory` 实时出行建议 | 航班/酒店/规划补充 | MCP 7 个工具 | 怎么去、带什么、天气/车次结论 |

第二批（依赖补工具）：`expense-claim` 报销凭证、`international-travel` 国际出行、`cost-optimization` 超标降本。

### 7.2 实现形态

采用"清单 + 按需注入 + 工具白名单 + 输出契约 + 版本"，而不是把技能正文全量塞进系统提示：

- **渐进披露**：目录（每个技能一行，始终在）→ 正文（激活时才注入，≤800 tokens）→ 参考文档（模型按需读取，并标注为数据而非指令）；
- **工具收敛**：窄技能用白名单砍掉 60%~80% 的 schema，宽技能保留全集 + `forbidden_tools`；
- **输出契约**：出现金额但无成功工具结果、出现"已预订"但无成功写操作，都记为可监控的违规；
- **不动路由**：`AgentRouter` 的四范式保持确定性，技能只负责注入手册与收敛工具。

### 7.3 部署与灰度

- 技能随镜像发布（`COPY skills ./skills`），版本跟镜像 tag 走，回滚 = 回滚镜像；
- ConfigMap 挂载 + `POST /api/v1/skills/reload` 做热修，全量校验通过才原子替换，失败保留上一版；
- 上线顺序：`SKILLS_SHADOW_MODE` 影子观察 → 按租户权重放量 → 全量；回滚有三层开关（全局 / 单技能 / 版本 pin）；
- 观测：`skill_activated_total`、`skill_contract_violation_total`、`skill_fallback_total`，trace 带 `skill.name` / `skill.version`。

落地分三个迭代：框架 + 3 个技能（3~4 人日）→ 规划/改签 + 契约校验 + 评测（4~5 人日）→ 第二批技能 + 租户灰度（5~6 人日）。验收标准：技能加载失败不影响启动；关闭 `SKILLS_ENABLED` 后行为与现在完全一致。

## 8. 可观测性

- OpenTelemetry 自动埋点（FastAPI/httpx）+ OTLP/Console exporter
- 手工 Span 覆盖：`chat.turn`、`intent.recognize`、`agent.route`、`memory.recall/compress`、`tool.invoke`、`llm.turn`、`confirm.execute`
- `X-Trace-Id` 贯通请求、审计与结构化日志；`/metrics` 暴露工具 P50、成功率、意图/范式分布、降级与预算触顶计数
- 审计事件带 `policy_version` / `prompt_version` / 工具参数与结果

## 9. 技术栈与选型

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

## 10. 目录

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
│  ├─ tools/travel_mcp.py      # 7 个 MCP 实时数据工具（可选注册）
│  ├─ prompt/travel_state.py   # 状态机与可用工具集
│  ├─ skills/                  # 技能层：加载 / 选择 / 契约（规划中，见第 7 节）
│  └─ harness/                 # 预算 / 确认门 / 降级
├─ domain/travel/              # 差旅模型、政策引擎、城市/机场参考数据
├─ infrastructure/mcp/         # MCP hub（HTTP/SSE）+ 实时数据解析与降级
├─ infrastructure/suppliers/   # 供应方适配层（当前 mock）
└─ infrastructure/observability/spans.py
skills/                        # SKILL.md 技能清单（规划中，见第 7 节）
docs/                          # 设计文档：04 差旅改造、06 MCP 集成、07 Skills 能力层
evals/                         # 离线评测（确定性 + 模型在环）
scripts/                       # 冒烟、无 pytest 测试运行器、MCP 健康检查
tests/golden/travel_dataset.py # 差旅金标门禁
```

## 11. 本地运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# 至少配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，以及 AUTH_DEV_TOKEN

docker compose up -d postgres redis etcd minio milvus-standalone
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

实时数据（可选）：在 `.env` 中配置 `MCP_12306_URL` / `MCP_GAODE_URL` / `MCP_BAZI_URL` / `MCP_FLIGHT_URL`（或一份 `MCP_SERVERS_JSON`），重启后自动注册 7 个 MCP 工具；不配置则这些工具不出现，其它功能不受影响。

```powershell
python scripts/check_mcp_health.py         # 连通性与工具列表；--json 供 CI 使用
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

## 12. API 示例

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

## 13. 评测口径

`python -m evals.run_eval --mode deterministic` 当前结果：

| 指标 | 结果 |
|---|---|
| 意图快车道 | 12/12 |
| 槽位解析 | 6/6 |
| 差标召回 | 6/6 |
| 审批链 | 5/5 |

这些是**规则与政策引擎**的确定性指标，不包含模型在环质量。轮次、无效工具调用率、token 与时延必须用 `--mode live` 在固定数据集、固定模型版本、`temperature=0` 下实测；仓库不预置任何性能结论。

## 14. 已知限制

- 供应方是 `MockTravelSupplier`，通过 `TravelSupplier` 协议抽象；接 TMC/航司时替换实现，**当前没有真实出票能力**
- 长期偏好写入需要 Milvus 可用；未连接时自动跳过，不影响主链路
- 金标只覆盖确定性部分；模型在环评测需要 API key 与固定数据集
- 采购域保留为参考实现，默认注册表、主入口与文档均以差旅为准
- `app/core/harness/fallback.py`、`app/core/prompt/templates.py` 仍是采购域遗留，差旅使用自己的降级与提示词
- 实时数据依赖外部 MCP 服务：未配置时相关工具不注册；服务不可用时返回明确的 `upstream_error`，不会伪造车次、票价或天气
- 技能层（Skills）目前是设计稿，运行时尚未加载；实现计划见 [docs/07-Skills能力层设计与落地方案.md](docs/07-Skills能力层设计与落地方案.md)

## 15. 设计文档

| 文档 | 内容 |
|---|---|
| [docs/04-差旅Agent改造说明.md](docs/04-差旅Agent改造说明.md) | 领域映射、四范式路由、差标引擎、预订安全、评测结果、与简历的对应关系 |
| [docs/06-实时数据MCP集成说明.md](docs/06-实时数据MCP集成说明.md) | MCP 传输、工具清单、解析与降级策略、配置与排障 |
| [docs/07-Skills能力层设计与落地方案.md](docs/07-Skills能力层设计与落地方案.md) | 技能选型、五种实现方案对比、代码改造点、部署与灰度 |
| [docs/01-记忆与Harness改造纪要.md](docs/01-记忆与Harness改造纪要.md) | 采购版历史文档：L0–L5 分层、P0→P2 实施明细 |
| [docs/02-完整Agent应用评估与补充路线.md](docs/02-完整Agent应用评估与补充路线.md) | 采购版历史文档：成熟度评估与补齐路线 |
| [docs/03-工具调用机制详解.md](docs/03-工具调用机制详解.md) | 采购版历史文档：工具调用生命周期 |

## 16. 压测

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
