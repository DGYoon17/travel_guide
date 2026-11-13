# 记忆管理与 Harness 改造纪要

> **历史文档（采购版）**：本文记录的是采购域改造过程。差旅主线的对应实现见
> [04-差旅Agent改造说明.md](04-差旅Agent改造说明.md)；两域共享同一套记忆与 harness 内核。

> **原始问题**：如果将这个项目进行改装，在记忆管理方面和 Harness 方面针对当前业务场景和应用背景来进行改装，那么会从哪些方面进行什么样的改进？
>
> 项目：ProcureMind AI 企业采购与审批助手
> 改造范围：记忆管理（L0–L5）与 Agent Harness
> 完成时间：2026-08-20
> 验证状态：112 个测试通过，pyflakes 干净，FastAPI app 与 OpenAPI 可正常构建，HTTP 层鉴权已实测

---

## 一、改造前的诊断

### 1.1 根本问题：两套骨架并存

代码里有两条互不连通的链路，这是排障时最费时间的一类问题：

| | 链路 A（真正跑着的） | 链路 B（写得更好但没接线） |
|---|---|---|
| 入口 | `main.py` → `api/routes/chat.py` | 无 |
| 编排 | `app/agent/orchestrator.py` `ProcurementOrchestrator` | `app/core/agent/orchestrator.py` `AgentOrchestrator` |
| 记忆 | `_maybe_summarize_thread` + `_trim_window`（无状态） | `ShortTermMemory`（进程内 dict） |
| 工具 | 硬编码 if/else 两个 | `ToolRegistry`（有 schema/discovery/MCP catalog） |

`grep` 证据：`long_term.py` 与 `summary.py` **零调用**；`ToolRegistry` 没有任何 API 层引用；`AgentOrchestrator` 只被测试用到。

### 1.2 记忆侧的 10 个问题

| # | 问题 | 后果 |
|---|---|---|
| 1 | 摘要做完就被丢掉：`_maybe_summarize_thread` 返回 `[摘要]+最后20条`（21 条），紧接着 `_trim_window` 取 `[-20:]` 把摘要裁走 | 白花一次 LLM 调用 |
| 2 | 阈值配置自相矛盾：`memory_summary_threshold=15 < memory_window_size=20` | 历史 16–20 条时用**空 transcript** 调 LLM |
| 3 | 服务端根本没有会话：`session_id` 只被原样回填进响应 | 多轮上下文全靠前端重传，"用户已确认的金额和审批人"由客户端持有并可篡改 |
| 4 | `ShortTermMemory` 用进程内 `dict[str, ShortTermMemory]` | 多 worker/多副本不共享，重启即丢；Redis 已连上但一个字节都没写 |
| 5 | 滑动窗口按单条 `popleft` | 会把 assistant 的 `tool_calls` 和对应 `role=tool` 拆开 → OpenAI 兼容接口直接 400 |
| 6 | `_trim_by_tokens` 每轮重算全量 token | O(n²) |
| 7 | `LongTermMemory` 与 RAG 文档共用同一 collection，`recall` 传 `intent_filter=None` | 查"用户记忆"会召回制度文档，反之亦然 |
| 8 | `_namespaced_id` 只影响写入 id，检索时**不按 namespace 过滤** | 多法人场景跨租户数据泄漏 |
| 9 | `remember` 无写入门槛、无 TTL、无遗忘、无 PII 拦截 | 一次性问句和敏感信息沉淀成"长期事实" |
| 10 | 采购必需的记忆维度全缺：组织画像、槽位状态、制度版本、事实留痕 | 用户每轮重复交代身份；模型靠猜填必填字段 |

### 1.3 Harness 侧的 13 个问题

| # | 问题 | 后果 |
|---|---|---|
| 1 | `stream_completion` 是伪流式：先 `await run_completion()` 拿完整文本再逐字符 yield | 首 token 延迟 = 全量延迟，SSE 白做 |
| 2 | 熔断器两份实现（`core/circuit_breaker.py` 与 `infrastructure/llm/circuit_breaker.py`） | 失败计数拆两份，阈值等于翻倍 |
| 3 | `_execute_tool` 硬编码 if/else，与 `ToolRegistry` 完全脱节 | 工具双轨 |
| 4 | 手写 95 行 function schema 与 pydantic 模型是两份真相 | 改一处另一处静默漂移 |
| 5 | `catalog` / `supplier` / `approval` 工具写完了**没暴露给模型** | 能力调不到 |
| 6 | `args["requester_id"]` / `_parse_date` 手写 split | 模型少传字段 → `KeyError`；填「下周五」→ `ValueError`；异常直接冒泡成 500 |
| 7 | **工具错误没有反馈闭环** | 用户看到接口报错，而不是 Agent 自我修正 |
| 8 | `submit_approval` 是真实写操作，但无确认门、无幂等、不可撤销、不留痕 | 不可逆后果无治理 |
| 9 | `PromptStateMachine` 用正则 `(确认\|同意\|可以\|…)` 推进到 `EXECUTING` | 用正则决定是否向 OA 提交上万元审批，"这个方案可以吗？"会命中 |
| 10 | `max_react_iterations` 触顶返回硬编码道歉并**丢弃全部工具输出** | 用户等了 5 轮，什么都没拿到 |
| 11 | 无 token/成本预算、无并发限流、无整体 deadline；熔断打开直接 `raise` → 500 | 有确定性兜底可用却不用 |
| 12 | `_setup_tracing()` 建了 TracerProvider 但**无 exporter、无埋点**；`react_trace` 塞进响应体 | 有可观测性代码、无可观测性；内部推理外泄 |
| 13 | `/chat` `/documents/ingest` **无鉴权**；`requester_id` 由 LLM 从对话里填；CORS `*` + credentials | **任何人可冒任何人身份发起请购单**；知识库开放写入 → 间接 prompt injection 通道 |

---

## 二、改造方案与实施

按"错误代价"而非时间远近重新分层，核心主张：

> 采购 Agent 的记忆主体不应该是 LLM 摘要出来的自由文本，而应该是**结构化、可校验、可追溯**的状态。
> 自然语言摘要只用于"用户偏好"这类错了也不致命的部分。

### 2.1 记忆分层（L0–L5）

| 层 | 内容 | 存储 | 关键约束 | 实现文件 |
|---|---|---|---|---|
| L0 工作记忆 | 本轮 tool_calls 轨迹 | 进程内 | 请求结束即弃 | `agent/orchestrator.py` |
| L1 会话记忆 | 消息 + 结构化滚动摘要 | Redis，租户+会话隔离 | 裁剪单位是消息组；摘要在不可裁剪的 pinned 区 | `core/memory/short_term.py`、`session_store.py`、`summary.py` |
| L2 槽位记忆 | `RequisitionDraft` 结构化草稿 | Redis，带版本号 | 已确认字段禁止模型静默改写 | `core/memory/draft.py` |
| L3 主体画像 | 部门/成本中心/职级/上级/预算编码 | HR/OA 主数据（只读） | 只从凭据与主数据来，模型不可见 | `core/memory/profile.py` |
| L4 组织知识 | 制度/SOP/供应商/案例 + 长期偏好 | Milvus，两个独立 collection | 带 `policy_version` 与租户过滤；只有 `active` 参与检索 | `core/memory/long_term.py`、`services/milvus_store.py` |
| L5 事实日志 | 谁在何时确认了什么 | PostgreSQL append-only | 只写不改，不可被摘要压缩 | `core/memory/audit.py`、`infrastructure/database/audit_repo.py` |

### 2.2 P0：正确性与安全（不做就是有洞）

#### P0-1 身份不可伪造

- 新增 `Principal`（不可变）与 `SubjectProfile`，`Scope` 做细粒度授权
- `requester_id` / `department` / `requester_level` / `tenant_id` 标记为 `injected_fields`，**从模型可见的 function schema 中剔除**（`schema.py` 的 `exclude`）
- 执行前由 `registry.parse_arguments()` 从凭据+主数据强制注入；模型若仍试图填写则丢弃并告警
- 副作用：`plan_purchase_requisition` 必填参数从 9 个降到 5 个，多轮补槽轮次自然减少

**验证**：模型传 `requester_id=E99999, department=财务部`，生成的请购单标题仍是"市场部"。

#### P0-2 鉴权、scope 与 CORS

- `require_principal` / `require_scopes(...)` 依赖；`StaticTokenResolver` 为可替换点（生产接 SSO/JWT introspect）
- **失败关闭**：没配任何 token 时接口返回 401，不退化成匿名可写；`AUTH_REQUIRED=false` 在生产直接 500
- 演示 token 在 `APP_ENV=production` 下被拒绝
- 权限分离：写入者不能自己发布知识库文档（`knowledge:write` ≠ `knowledge:publish`）
- CORS 由 `*` 改显式白名单（`*` + `allow_credentials=True` 浏览器本就拒绝）

**实测**：无 token→401、错 token→401、缺 `metrics:read`→403、缺 `knowledge:write`→403。

#### P0-3 工具层重建与错误回喂闭环

两层策略：

1. **能解析的解析掉**（`coercion.py`）——这才是真正减少失败轮次的办法
   - 日期：`2026-09-15` / `2026年9月15日` / `20260915` / `今天` / `明天` / `下周五` / `本周三` / `7天后` / `3个工作日后`（跳周末）/ `2周后` / `1个月后` / `本月底` / `下月初` / `1月10日`（已过则滚到明年）
   - 金额：`15000` / `15,000元` / `￥15000` / `1.5万` / `1万5` / `8千` / `18.8k` / `人民币 6299 元`
   - 数量：`3` / `"3"` / `"3个"` / `3.0`，拒绝小数、越界、布尔
2. **真解析不了才回报字段级错误**（`errors.py`）
   - `ToolFailure.to_observation()` 输出 JSON：`error_code` + `field_issues[{field, problem, expected, example}]` + `retry_allowed` + `next_step`
   - `REPAIRABLE_CODES` 区分可修复（缺参/参数错/JSON 坏/未知工具/前置条件未满足）与不可修复（权限/超时/下游异常）
   - 可修复 → 回喂模型自修正，最多 `TOOL_MAX_REPAIR_ATTEMPTS` 轮后转为向用户提问
   - 不可修复 → 立即停止，不让模型反复试

同时：`ToolSpec` 成为单一真相（schema 由 pydantic 派生并内联 `$ref`、折叠 `anyOf[T,null]`）、per-tool 超时、只读工具**并发执行**（改造前是顺序 for-await）、未预期异常一律转 `ToolFailure` 绝不冒泡。

**验证**：第 1 轮漏填 → 结构化错误回喂 → 第 2 轮补齐 → 成功，全程无 500。

#### P0-4 记忆三个 bug 的结构性修复

| 问题 | 修法 |
|---|---|
| 摘要被裁掉 | 摘要移到独立的 `_pinned` 区，**不参与裁剪** → 物理上不可能被截断 |
| `tool_calls` 被裁断 | 裁剪单位改为 `_Group`（assistant+tool_calls 与其全部 tool 结果绑定，不可拆）；未收齐结果的残缺组不发给模型；孤立 tool 结果直接丢弃 |
| O(n²) token 计数 | 每条消息缓存 token 数，组级汇总，增删时增量维护 |
| 阈值倒挂 | 用 `memory_summary_trigger_tokens < memory_max_tokens` 替代原先的条数阈值；未触发时不调 LLM |
| 最后一组被裁 | `_trim` 保护最后一组，避免丢掉当前轮的用户输入 |

#### P0-5 写操作确认门 + 幂等 + 审计

- 工具分 `READ` / `DRAFT` / `WRITE` 三级，只有 `WRITE` 走确认门
- 模型调用 WRITE 工具时**不执行**：先校验参数（失败仍走可修复错误）→ 检查前置条件 → 生成确定性预览 → 登记 `PendingAction` → 审计 `ACTION_PENDING` → 中断本轮
- 回给模型的观察值是 `CONFIRMATION_REQUIRED`，明确"不要声称已提交，不要给单号"
- 回给用户的文案是**确定性拼装的预览**，不让模型润色——摘要里的金额与审批链一旦被改写，用户确认的就不是实际要提交的那份
- 真正执行只能通过 `POST /api/v1/approvals/confirm` 带 `action_id`，**不经任何自然语言判定**
- 幂等键 = `租户:主体:请购单ID:草稿内容指纹`；审批单号由幂等键确定性派生（不再 `uuid4()`）
- 确认前草稿被改过 → 指纹不符 → 409 拒绝，要求重新生成
- 他人不能确认自己的动作（403）；超时 410
- 正则推进 `EXECUTING` 的授权路径已移除，`PromptStateMachine` 降级为纯提示词策略，`EXECUTING` 只能由 `mark_executing()` 显式置入

### 2.3 P1：能力补齐

- **P1-6** Redis 落地 L1+L2，`session_id` 真正生效，前端只需传本轮 `message`；会话与主体绑定（猜到 id 也读不到别人的上下文）；版本号 CAS 防并发覆盖
- **P1-7** 工具统一到 registry，`catalog` / `supplier` / `knowledge` / `approval` 全部接入；`policy_query.py` 改为委托领域层，消掉审批路径的第二份真相
- **P1-8** 真流式：`LLMService.stream_turn` 按 `index` 累积 `tool_calls` 分片；SSE 事件扩为 `status` / `tool_call` / `tool_result` / `content` / `confirmation_required` / `done` / `error`
- **P1-9** 删掉未接线的 `AgentOrchestrator`；合并双熔断器为共享单例，并补 `record_external_failure()`（流式失败发生在 `call()` 之外，原本永远不计入熔断）；清掉 `travel` 别名与空目录

### 2.4 P2：工程成熟度

- **P2-10** L4 独立 collection + 强制租户/主体/过期/制度版本过滤 + 写入判别（PII 与一次性表述拦截）；L5 审计 append-only（只暴露 insert/select）；摘要改**结构化 + 数字可追溯**——摘要里的每个金额/日期必须能在对话原文或 L2 草稿中溯源，否则丢弃并告警
- **P2-11** 四重预算（轮次/工具调用数/token/墙钟 deadline）；触顶时用已有工具产出 best-effort 收尾（保留请购单内容与缺失项追问），不再丢弃；LLM 不可用时走 `fallback.py` 的规则降级，给出确定性审批链与阈值并标注降级
- **P2-12** OTel exporter（OTLP/Console）+ FastAPI/httpx 埋点 + `X-Trace-Id` 贯通；`/metrics` `/audit/recent`；`react_trace`/`plan_steps` 不再回传客户端；金标评测集入 CI

---

## 三、顺手修掉的既有 bug

改造过程中发现并修复的、与原议题无关但会在运行时爆炸的问题：

| 问题 | 位置 | 后果 |
|---|---|---|
| `ZoneInfo("Asia/Shanghai")` 无 tzdata 直接抛异常 | `requisition.py` 函数体内构造 | Windows / 精简镜像上生成请购单时才崩；已抽出 `core/clock.py` 并回落固定 +08:00 |
| embedding 维度 1536 vs 模型默认 1024 | `milvus_store.py` 硬编码 | 入库必然失败；改为配置项并显式传 `dimensions`，提前校验 |
| `chunk_text(text, chunk_overlap=...)` 参数名与签名 `overlap` 不符 | `etl/pipeline.py` | 调用即 `TypeError` |
| `policy_query` 与领域层两套审批链 | `tools/policy_query.py` | 用户拿到的审批路径取决于走了哪条代码路径；已统一委托领域层 |
| 依赖不可用被报成健康（`redis is None` → `checks["redis"]=True`） | `routes/health.py` | Redis 挂掉探针显示 ok，而会话已退化为进程内存；已区分连通/降级/未配置，并补 `/health/live` |
| docker-compose 设 `OPENAI_API_KEY`，代码读 `DASHSCOPE_API_KEY` | `docker-compose.yml` | 容器里永远拿不到 key |
| Dockerfile 以 root 运行 | `Dockerfile` | 已加 uid 10001 非 root 用户 |
| `.env.example` 提交了真实格式的 `sk-` 密钥 | `.env.example` | 已替换为占位值（**原密钥需吊销**） |

---

## 四、交付物清单

### 4.1 新增模块（19 个）

```
app/core/clock.py                        业务时区，tzdata 缺失时回落 +08:00
app/core/security/principal.py           Principal / Scope
app/core/security/auth.py                Bearer 鉴权、scope 依赖、失败关闭
app/core/memory/profile.py               L3 主体画像与 ProfileProvider
app/core/memory/draft.py                 L2 槽位记忆 RequisitionDraft
app/core/memory/session_store.py         L1+L2 持久化，Redis CAS
app/core/memory/audit.py                 L5 事实日志与脱敏
app/core/harness/budget.py               四重运行预算
app/core/harness/confirmation.py         确认门与幂等存储
app/core/harness/fallback.py             规则降级答复
app/core/tools/spec.py                   ToolSpec / ToolAccess / ToolContext
app/core/tools/schema.py                 pydantic → function schema
app/core/tools/errors.py                 结构化工具错误
app/core/tools/coercion.py               日期/金额/数量宽松解析
app/core/tools/builtin.py                内置工具统一注册
app/infrastructure/database/audit_repo.py 审计表（append-only）
app/api/routes/approvals.py              写操作唯一执行入口
app/api/routes/observability.py          /metrics /audit/recent
docs/                                    本文档与评估报告
```

### 4.2 重写模块（12 个）

`config.py`、`main.py`、`agent/orchestrator.py`、`core/agent/orchestrator.py`、`core/memory/short_term.py`、`core/memory/summary.py`、`core/memory/long_term.py`、`core/tools/registry.py`、`core/tools/policy_query.py`、`core/tools/approval.py`、`services/llm.py`、`services/milvus_store.py`、`services/embeddings.py`、`domain/schemas.py`、`api/routes/chat.py`、`api/routes/documents.py`、`api/routes/health.py`、`README.md`

### 4.3 删除

`app/core/circuit_breaker.py`（重复实现）、`app/domain/travel/`（空目录）、`AgentOrchestrator` 及其私有适配器类、`Travel*` 别名

### 4.4 测试（112 个）

| 文件 | 数量 | 覆盖 |
|---|---|---|
| `test_coercion.py` | 34 | 日期/金额/数量解析与错误引导 |
| `test_orchestrator_harness.py` | 17 | 会话状态、错误闭环、确认门、幂等、预算收尾、降级、流式 |
| `test_tool_registry.py` | 16 | 身份注入、schema 派生、错误回喂、L2 回填、注入防护 |
| `test_session_and_auth.py` | 11 | 会话隔离、乐观锁、token 解析、失败关闭 |
| `test_draft.py` | 9 | 已确认字段保护、缺失追问、指纹稳定性 |
| `test_short_term_memory.py` | 7 | 摘要不被裁、消息组不可拆、增量计数 |
| `test_summary.py` | 6 | 结构化摘要、数字可追溯、失败回退 |
| `test_golden_eval.py` | 6 | 金标门禁（意图/参数/合规/审批链） |
| `test_intent.py` / `test_rag.py` / `test_orchestrator.py` | 6 | 既有测试 |

**金标评测实测**（`tests/golden/dataset.py`，阈值化门禁）：

| 指标 | 实测 | 阈值 | 说明 |
|---|---|---|---|
| 意图分类准确率 | 90.91%（10/11） | ≥80% | 1 例真实失败：「公司内部的报销制度手册在哪」判成 `info_query` 而非 `rag` |
| 参数抽取准确率 | 100%（6/6） | =100% | 抽错直接导致错误审批链，不允许回退 |
| 合规召回率 | 100% | =100% | 漏报零容忍 |
| 审批路径正确率 | 100% | =100% | 含金额单调性检查 |

---

## 五、验证方式与结果

```
pytest                       →  112 passed
python -m pyflakes app tests →  clean
python -m compileall app tests → OK
app.openapi()                →  10 个路由全部注册，响应模型合法
```

HTTP 层手工冒烟（TestClient，**未落为提交的测试**）：

```
POST /api/v1/chat        无 token            → 401 缺少 Bearer 访问令牌
POST /api/v1/chat        错误 token          → 401 访问令牌无效或已过期
GET  /api/v1/metrics     缺 metrics:read     → 403 缺少所需权限：metrics:read
POST /api/v1/documents/ingest 缺 knowledge:write → 403
GET  /api/v1/health      依赖全不可用        → 200 degraded，checks 如实反映
GET  /api/v1/health/live                     → 200
响应头 X-Trace-Id                            → 存在
```

---

## 六、需要立即处理的两件事

1. **吊销泄漏的密钥**。`.env.example` 原本提交了一个真实格式的 `sk-111f7cc6...`，`.env` 里是同一把。已替换为占位值，但若该 key 曾有效，请在 DashScope 控制台吊销重签。
2. **重建 Milvus collection**。知识库 schema 新增了 `status` / `tenant_id` / `policy_version` / `effective_from`，对已存在的 collection 不生效；长期记忆使用新的独立 collection（`ensure_memory_collection`）。

---

## 七、本次改造明确没做的部分

| 项 | 状态 | 说明 |
|---|---|---|
| catalog / supplier / approval 下游 | 进程内模拟 | 接 SRM/OA 时替换实现，`idempotency_key` 直接作为下游 `Idempotency-Key` |
| L4 长期记忆在线流程 | 类已就绪，未接编排器 | 写入判别与召回注入 prompt 的策略未实现 |
| 模型在环的端到端评测 | 未做 | 幻觉率、指令遵循率需另建离线任务，不适合放单元测试 |
| `ReActAgent` / `PlanningAgent` / `ReflectionAgent` | 文件保留，未启用 | 主链路走原生 function calling；写操作自检用 `LLMService.complete` 直接实现，未复用 `ReflectionAgent` |
| `MilvusDocumentStore.set_status` | 删旧插新 | 大批量发布应改走离线重建 |

后续完善方向见 `docs/02-完整Agent应用评估与补充路线.md`。
