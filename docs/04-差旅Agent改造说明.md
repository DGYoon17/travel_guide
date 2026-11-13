# 差旅 Agent 改造说明

本文记录"把采购助手改造成企业差旅助手，并让简历里的每条描述都有代码与评测对应"的具体落地。
采购域代码保留为同一套内核的参考实现，主入口、默认工具注册表与文档均以差旅为准。

---

## 一、领域迁移

| 采购域（改造前） | 差旅域（现在） |
|---|---|
| `RequisitionDraft` | `TripDraft`：出发地/目的地/日期/舱位/住宿/预算/证件 |
| `ProcurementPolicy` | `TravelPolicy`：职级 × 城市等级 × 舱位/住宿标准、提前天数、受限目的地、审批阈值 |
| `PurchaseRequisition` | `Itinerary` + `ItinerarySegment` + `TravelOffer` |
| 请购单提交审批 | 机票/酒店**组合预订** + 出差审批 |
| 供应商准入 | 协议价/差标校验 |
| 6 个采购工具 | 11 个差旅工具 + 可选 7 个 MCP 实时数据工具 |

关键文件：

- `app/domain/travel/models.py`：领域模型
- `app/domain/travel/policy.py`：确定性差标引擎
- `app/core/memory/trip_draft.py`：L2 行程草稿
- `app/core/tools/travel_builtin.py`：工具集
- `app/agent/travel_orchestrator.py`：主编排

---

## 二、四范式路由

`app/core/agent/router.py`

| 范式 | 触发 | 执行 |
|---|---|---|
| `DIRECT` | 差标/制度问答、闲聊 | 单工具 + 一次生成 |
| `REACT` | 槽位不全、预订/审批、单域检索 | function calling 循环 |
| `PLAN_EXECUTE` | 槽位齐全的规划请求 | 固定模板计划 + 顺序执行 |
| `CLARIFY` | 置信度 < 0.85 或意图未知 | 不调工具，先澄清 |

设计取舍：

1. **路由用确定性规则，不让 LLM 选范式**。可单测、可解释、可回归；LLM 只做意图分类。
2. **槽位不全时不进 PLAN_EXECUTE**。模板计划没有参数可填，硬跑只会得到确定性追问；此时走 ReAct 让模型从对话里抽槽。抽槽通过工具的 `draft_updates` 回写 L2。
3. **写操作永远走 ReAct + 确认门**，即使行程要素齐全，也不塞进固定流水线。
4. 意图识别新增 `UNKNOWN`：规则与模型都没给出可信意图时显式标记，路由进 `CLARIFY`；不再用 `GENERAL` 兜底把"安排一下"当成闲聊。

---

## 三、差标引擎

`app/domain/travel/policy.py`

- 舱位：职级 × 国内/国际 × 飞行时长；长航线可放宽一档
- 住宿：一线/新一线/其他 × 职级，按每晚上限算超标金额
- 提前预订：国内 3 天 / 国际 7 天
- 硬约束：证件未就绪、目的地受限、日期倒置 → `block`
- 审批链：金额 → 财务 BP / 分管高管；国际 → 合规审核；超标 → 费用合规复核

每个结论带 `code`、`expected`、`over_standard_cny`。模型只转述，不解释制度。

---

## 四、预订安全

1. `plan_trip` 生成 `last_plan`（航段、总价、审批链、有效期）；
2. `book_trip` 只创建 `PendingAction`，不执行；
3. 确认门把每个 `offer_id` 的**报价快照**写进 `PendingAction.metadata`；
4. `/approvals/confirm` 重新 `quote`，价差 > 2% 返回 `409`，删除待确认动作；
5. 组合预订任一航段失败 → 取消已成功订单（补偿），错误里列出已回滚订单；
6. 身份字段在执行时重新注入，确认门存储的 `arguments` 已剔除这些字段。

对应测试：

- `test_booking_is_gated_and_only_confirm_executes`
- `test_price_change_aborts_confirmation`
- `test_book_trip_creates_bookings_and_cancel_works`

---

## 五、记忆、追踪与 Provider

### L4 长期偏好

`app/core/memory/travel_memory.py` 按租户/主体装配 `LongTermMemory`，`main.py` 在 Milvus
可用时注入编排器；编排器在系统提示里以 `<memory>` 数据块注入，并只对"我习惯/我偏好/以后都"
这类明确表述写入。

### 手工 Span

`app/infrastructure/observability/spans.py` 封装 OTel；`TravelOrchestrator` 在
`chat.turn` / `intent.recognize` / `agent.route` / `memory.recall` / `memory.compress` /
`tool.invoke` / `llm.turn` / `confirm.execute` 建 Span，并记录 `paradigm`、`route.reason`、
`tool`、`error_code`、`action_id` 等属性。

### 双 Provider

`app/config.py` 增加 `LLM_PROVIDER=dashscope|openai`，`llm_api_key/base/model` 按供应方取值。
`LLMService` 与 `LLMClient` 仍走 OpenAI 兼容协议，切换供应商不需要改调用方。

---

## 六、评测结果

确定性指标（`python -m evals.run_eval --mode deterministic`）：

| 指标 | 结果 |
|---|---|
| 意图快车道 | 12/12 |
| 槽位解析 | 6/6 |
| 差标召回 | 6/6 |
| 审批链 | 5/5 |

测试：

- 新增 42 项差旅测试（政策、路由、工具、编排、金标）
- 新增 73 项实时数据（MCP）测试：传输（HTTP/SSE）、解析与降级、工具行为、PLAN_EXECUTE 端到端
- 离线运行器 `python scripts/run_tests_lite.py`：231 通过，0 失败
- 采购旧测试里依赖"下周五"的用例已固定到参考日，避免随时间腐烂

模型在环指标（轮次、无效工具调用率、token、时延）必须用：

```powershell
python -m evals.run_eval --mode live --router on
python -m evals.run_eval --mode live --router off
```

在固定数据集、固定模型版本、`temperature=0` 下实测。仓库不预置这些数字。

---

## 七、与简历描述的对应关系

| 简历原句 | 现在是否成立 | 证据 |
|---|---|---|
| 企业差旅场景，行程咨询/差标解读/预订辅助 | ✅ | 11 个差旅工具（+ 可选 7 个 MCP 实时数据工具）、差标引擎、预订确认门 |
| 多模式 Agent 编排、动态路由范式 | ✅ | `app/core/agent/router.py` 四范式 + 单测 |
| 低置信度触发校验 | ✅ | `UNKNOWN` + 0.85 阈值 + `CLARIFY`，阈值两侧可达 |
| 工具失败触发校验/自修正 | ✅ | 字段级 `field_issues` 回喂，最多 2 轮 |
| 离线评测平均推理轮次 3.0→1.73 | ❌ 未实测 | 需要 `--mode live --router on/off` 报告 |
| 无效工具调用率下降 35% | ❌ 未实测 | 指标已在 `evals/run_eval.py` 定义，缺基线 |
| 意图识别降低延迟 30% | ❌ 未压测 | 快车道只省一次分类调用，不能归因到端到端 |
| 长对话 token 下降 29% | ❌ 未实测 | 摘要与 pinned 区已实现，缺对比报告 |
| 各阶段 Span、trace_id 还原 | ✅ | `spans.py` + 编排器 8 类 Span |
| 稳定事实向量化写入 Milvus | ✅ | `TenantLongTermMemory` 已接 `main.py` |
| OpenAI API | ✅ 可切换 | `LLM_PROVIDER=openai`，默认 DashScope |
| PostgreSQL | ✅ | L5 append-only 审计 |

**结论**：架构类描述已经落地；性能类数字仍必须用 `evals/` 的 live 模式测出来后才能写。
没有报告支撑的数字，建议从简历删除或改成定性描述。

---

## 八、边界

- `MockTravelSupplier` 是 mock，不是真实 TMC/航司；通过 `TravelSupplier` 协议替换
- L4 依赖 Milvus；未连接时自动跳过，不影响主链路
- `app/core/harness/fallback.py` 与 `app/core/prompt/templates.py` 仍是采购域遗留，差旅使用
  `TravelOrchestrator._degraded_reply` 与自己的系统提示
- 采购域测试保留，用于验证共享内核没有退化

---

## 九、实时数据（MCP）增量

在 11 个差旅工具之外新增 7 个**只读**实时数据工具：12306 车次（自动站点代码）、高德自驾
（距离/耗时/过路费）、高德酒店（按预算筛选）、高德天气（多日预报）、八字黄历、航班（长途自动触发）、
城际交通对比。它们仅在配置了 MCP 服务时注册，不参与差标与预订；PLAN_EXECUTE 在方案生成后追加查询，
结果写 L2 快照供后续轮次引用，服务不可用时按 `upstream_error` 如实降级。

协议实现、解析策略、配置与排障见 `docs/06-实时数据MCP集成说明.md`。
