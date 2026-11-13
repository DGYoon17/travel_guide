# 实时数据 MCP 集成说明

本文说明如何把参考项目（`travel-planning-agent`）里基于 MCP 的实时数据能力迁移进本仓库，
以及迁移时做了哪些工程化取舍：**协议自己实现、降级必须可标注、外部服务不可靠是常态**。

---

## 一、范围与目标

参考项目通过 SSE 连接四个 MCP 服务，为旅行规划补充实时事实：

| 能力 | 参考实现 | 本仓库对应 |
|---|---|---|
| 12306 火车票（自动站点代码） | `nodes.train_query_node` | `query_train_tickets` + `LiveTravelData.train_tickets` |
| 高德自驾（距离/时间/过路费） | `train_query_node` 内联逻辑 | `plan_driving_route` + `LiveTravelData.driving_route` |
| 高德酒店（按预算筛选） | `rag_search_node` 酒店搜索 | `search_local_hotels` + `LiveTravelData.hotel_search` |
| 高德天气（多日预报） | `weather_query_node` | `query_weather_forecast` + `LiveTravelData.weather` |
| 八字黄历（农历/宜忌） | `lucky_day_query_node` | `query_lunar_almanac` + `LiveTravelData.almanac` |
| 航班（>800km 自动触发） | `action_node` 内 IATA 映射 | `query_flights` + `LiveTravelData.flights` + 距离触发 |
| 车次 + 自驾组合 | `train_query` 复合工具 | `plan_intercity_transport` |

目标不是"抄一份调用代码"，而是让它在现有 harness 里成立：

1. 工具必须是**声明式注册**（pydantic 参数模型 → function schema），不能手写 prompt 里的工具表；
2. 外部失败必须变成**模型能读懂的观察值**，不能抛栈、不能让整轮 500；
3. 不能因为实时查询把方案生成与最终答复的时间预算吃光；
4. 未配置 MCP 的部署（单测、离线评测）必须完全不受影响。

---

## 二、代码结构

```text
app/
├─ core/tools/mcp_client.py          # 传输层：MCPHttpClient（HTTP JSON-RPC）+ MCPSseClient（SSE）
├─ infrastructure/mcp/
│  ├─ hub.py                         # 服务注册、传输选择、超时/重试/重建连接、健康检查
│  └─ travel_data.py                 # 业务门面 LiveTravelData + 宽容解析 + 降级策略
├─ core/tools/travel_mcp.py          # 7 个 ToolSpec：参数模型、渲染、错误翻译、注册
└─ domain/travel/geo.py              # 城市坐标 / 机场三字码 / 距离估算（只读参考数据）

scripts/check_mcp_health.py          # 连通性 + 工具列表；--json 供 CI
tests/mcp_fakes.py                   # FakeMCPHub + 标准响应
tests/test_mcp_client.py             # SSE 握手/配对、HTTP 客户端、hub 重试、配置解析
tests/test_mcp_travel_data.py        # 解析器 + 门面降级
tests/test_mcp_tools.py              # 工具行为、预算筛选、自动航班、编排接入
```

分层原则：

- **传输层不认识业务**：只认 JSON-RPC 方法与 `content` 文本块；
- **hub 不认识差旅**：只认 `server + tool + arguments`，负责可靠性与生命周期；
- **门面认识业务但不认识模型**：把返回归一化成 `TrainTickets` / `DrivingRoute` / `WeatherReport` 等结构；
- **工具层负责"给模型看什么"**：渲染文本、截断、错误翻译、写 L2 快照。

---

## 三、协议与传输实现

### 1. HTTP JSON-RPC（`transport=http`）

`POST <url>`，`Content-Type: application/json`，`Accept: application/json, text/event-stream`；
响应可能是普通 JSON，也可能是 `text/event-stream` 包裹的单条结果（部分网关如此），两种都解析。

### 2. SSE（`transport=sse`，URL 通常以 `/sse` 结尾）

经典 MCP SSE 不是"POST 完就返回"，而是：

1. `GET /sse` 建立长连接（`Accept: text/event-stream`）；
2. 服务端先发 `event: endpoint`，`data` 是消息投递地址（相对路径时用 `urljoin` 补全）；
3. 客户端把 JSON-RPC 请求 `POST` 到该地址，服务端通常回 `202 Accepted`；
4. 响应**异步回到 GET 事件流**，按 JSON-RPC `id` 与 pending future 配对；
5. `close()` 取消读流任务并把所有未完成请求置为失败，避免协程悬挂。

除握手之外，两种传输都会在 `initialize` 成功后发送 `notifications/initialized` 通知：
部分服务端会拒绝"未通知就调用"的请求，而这个通知失败不会阻塞主流程（只记日志）。

这也是最容易踩坑的地方：只发 POST 不读流，会永远等不到结果；读流不配对 id，会在并发调用时串结果。

### 3. 可靠性

| 问题 | 处理 |
|---|---|
| 连接中断 / 超时 | 有限重试（`MCP_MAX_RETRIES`，默认 2，指数退避 0.4s×n），丢弃旧连接重建 |
| JSON-RPC 协议错误 | **不重试**：同样的参数再发一次只会再错一次，直接返回可读错误 |
| `isError: true` 的工具结果 | 视为业务错误，不重试，文本透传给工具层 |
| 服务未配置 | `MCPError("未配置 MCP 服务器 …")`，不发起网络请求 |
| 服务不可用 | 工具返回 `upstream_error`，hint 明确"不要编造车次/航班/票价/天气" |

重试只发生在传输层：`asyncio.TimeoutError` 与 `MCPTransportError`。业务解析失败（例如拿不到距离）
不重试，直接走降级分支。

---

## 四、工具清单

所有工具都是 `ToolAccess.READ`，不进确认门，无 scope 要求。

| 工具 | 参数（均可缺省，从 L2 回填） | 行为 |
|---|---|---|
| `plan_intercity_transport` | `origin` / `destination` / `date` / `include_flight` | 火车 + 自驾 +（长途）航班，给出确定性建议 |
| `query_train_tickets` | `origin` / `destination` / `date` | 站点代码 → `get-tickets`，失败退回城市名 |
| `plan_driving_route` | `origin` / `destination` | `maps_geo` ×2 → `maps_direction_driving`，解析距离/耗时/过路费 |
| `search_local_hotels` | `city` / `budget_per_night_cny` / `star` / `keywords` | 预算 → 关键词 → POI 搜索 → 价格二次过滤 |
| `query_weather_forecast` | `city` / `start_date` / `days` | 按行程日期逐日匹配，缺失日期明确说明；`days` 缺省按行程天数 |
| `query_lunar_almanac` | `date` | `getChineseCalendar`（先 `solarDatetime`，再 `date` 兜底） |
| `query_flights` | `dep_city` / `arr_city` / `date` | 城市名 → IATA，再 `searchFlightsByDepArr` |

缺参时的回填顺序：**工具参数 → L2 草稿（`origin_city`/`destination_city`/`depart_date`/`budget_cny`）→ 字段级追问**。
这样模型漏填不会变成"你要从哪出发"的反问，也不会让工具拿空参数去请求外部服务。

---

## 五、解析与降级策略

### 1. 站点代码

顺序：`get-station-code-of-citys`（`citys`）→ `get-stations-code-in-city`（`city`）→ 同工具 `cities` 参数 → 城市名直接查车次。

- 城市名归一化：剥离「市/特别行政区/自治区」等后缀；
- 支持四种返回形状：按城市分组的 dict、`data` 包裹的 list、扁平 list、纯文本 `上海: SHH`；
- 解析出的站点代码缓存 `MCP_STATION_CODE_TTL_SECONDS`（默认 24h），失败**不缓存**；
- 拿不到代码不是硬失败：改用城市名查询并把"站点代码未解析"写进 `notes`。

### 2. 自驾路线

- 距离/耗时按高德原始字段解释：`distance` 米、`duration` 秒、`tolls` 元；同时兼容 `distance_km` / `duration_minutes` / `176.4公里` 这类已带单位的返回；
- `paths` 语义是**备选路线**，取最短的一条，不做求和（求和会把里程翻倍）；
- 解析不出距离或高德不可用 → 退化为"城市中心点直线距离 ×1.25"，标记 `estimated=true`，
  **过路费一律不给**（编一个过路费比不给更糟）；
- 超过 300km 提示疲劳驾驶，超过 500km 建议改用高铁/飞机。

### 3. 酒店

两层筛选：

1. 关键词按每晚预算分档：`>500` 高端/豪华酒店、`>300` 酒店/品牌酒店、其余经济型/民宿；
2. 解析出 `biz_ext.cost` 等效价格后过滤超预算项，并在文案里说明过滤了几家。

每晚预算来源：显式参数 → `budget_cny ÷ 住宿晚数` → 不限。全部超预算时保留原列表并标注，
而不是返回空（用户可能愿意微调预算）。

### 4. 天气

- 解析 `forecasts` / `casts` 列表，字段兼容中英文（`dayweather`/`weather`、`daytemp`/`high`）；
- 有行程日期时逐日精确匹配，缺失日期进 `missing_dates`，文案提示"超出当前预报范围，建议出行前 1-2 天再查"；
- 无行程日期时按 `days` 取前 N 天。

### 5. 黄历

服务商字段差异最大（`公历/农历/干支/宜/忌` vs 英文键），因此遍历嵌套 dict 选"命中字段最多"的一层，
再归一化。`宜/忌` 中出现"出行/出门"时给出明确结论，否则给"无特别宜忌"。黄历只作文本参考，
不参与差标与预订判断。

### 6. 航班

城市名 → IATA 用本地只读表（`app/domain/travel/geo.py`），支持"上海虹桥"这类带后缀的输入；
查不到时返回字段级错误（`INVALID_ARGUMENTS` + `dep_city`/`arr_city`），让模型改参数，而不是静默跳过。

自动触发规则（`settings.flight_auto_trigger_km`，默认 800）：

| 条件 | 行为 |
|---|---|
| `include_flight=false` | 不查，理由写进结果 |
| 未配置 `flight Server` | 不查，理由写进结果 |
| 距离未知 | 查（交给航班结果判断） |
| 距离 ≥ 800km | 查 |
| 距离 < 800km | 不查，理由写进结果 |

距离来自城市坐标估算，**明确标注"估算距离，真实里程以高德/12306 为准"**。

---

## 六、编排接入

`TravelOrchestrator` 在 PLAN_EXECUTE 的计划末尾追加实时步骤（仅当 `ctx.mcp is not None`）：

```text
get_travel_policy → search_flights → search_hotels → check_travel_compliance → plan_trip
  → plan_intercity_transport → search_local_hotels（需要住宿时）→ query_weather_forecast → query_lunar_almanac
```

顺序与预算的取舍：

- 实时数据放在 `plan_trip` 之后：它不参与总价与审批链，MCP 全挂也要先把方案交出去；
- 每个实时步骤有独立 deadline：`剩余时间 - 20s`（预留给最终答复生成），超时转成 `TIMEOUT` 观察值；
- 剩余时间 `<25s` 直接跳过实时步骤，并记录 `plan.realtime_step_skipped`；
- 复合工具内部**并发**查询车次/自驾/航班（各自 65s / 25s / 25s 预算），总耗时约等于最慢一项，
  单项超时或失败只影响该分项，不会把整次查询判死。

结果落 L2 草稿快照（`last_transport` / `last_weather` / `last_almanac` / `last_hotels`），
系统提示里以"本会话已查询到的实时数据"注入。短期记忆会被摘要裁剪，快照保证后续轮次仍然知道
"查过什么、结论是什么"，不会重复查询或凭空重述。

---

## 七、配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MCP_ENABLED` | `true` | 关闭后不注册任何实时数据工具 |
| `MCP_TRANSPORT` | `auto` | `auto`：URL 以 `/sse` 结尾走 SSE，其余 HTTP |
| `MCP_12306_URL` | 空 | 12306 服务（必需能力的来源） |
| `MCP_GAODE_URL` | 空 | 高德服务（必需能力的来源） |
| `MCP_BAZI_URL` | 空 | 黄历服务（可选） |
| `MCP_FLIGHT_URL` | 空 | 航班服务（可选） |
| `MCP_SERVERS_JSON` | 空 | 覆盖上面四个字段；数组或 `{"mcp_servers": [...]}` |
| `MCP_SERVERS_CONFIG_PATH` | 空 | 指向参考项目的 `servers_config.json`，便于直接迁移 |
| `MCP_TIMEOUT_SECONDS` | `25` | 单次请求超时（秒） |
| `MCP_MAX_RETRIES` | `2` | 传输层重试次数 |
| `MCP_STATION_CODE_TTL_SECONDS` | `86400` | 站点代码缓存 |
| `FLIGHT_AUTO_TRIGGER_KM` | `800` | 自动查航班的距离阈值 |

示例（`MCP_SERVERS_JSON`）：

```json
[
  {"name": "Gaode Server", "url": "https://example.com/gaode/sse", "transport": "sse",
   "headers": {"X-API-Key": "..."}},
  {"name": "12306 Server", "url": "https://example.com/12306/sse"},
  {"name": "bazi Server", "url": "https://example.com/bazi/sse"}
]
```

未配置任何 URL 时 `build_mcp_hub` 返回 `None`：注册表不包含这 7 个工具，编排计划也不追加实时步骤，
其它能力（差标、方案、预订、审计）完全不受影响。参考项目示例里的 `https://your-xxx-mcp-server-url/sse`
占位符会被识别并跳过，不会被注册成"必定失败"的工具。

---

## 八、排障

```powershell
python scripts/check_mcp_health.py          # 人类可读
python scripts/check_mcp_health.py --json   # CI / 监控
python scripts/check_mcp_health.py --no-tools  # 只探连通性
```

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `SSE 握手超时` | URL 不是 SSE 端点，或网关要求鉴权头 | 确认 URL 以 `/sse` 结尾；用 `headers` 传 Key |
| `404` 且工具名不存在 | 服务端工具名与参考项目不同 | 用 `check_mcp_health.py` 看实际工具名，必要时扩展 `LiveTravelData` 的候选工具名 |
| 站点代码一直查不到 | 城市名与 12306 站点命名不一致 | 车次仍会用城市名兜底；必要时扩展 `geo.py` 参考表 |
| 天气"暂无预报" | 超出预报窗口（通常 3~7 天） | 属于正常降级，文案已提示出行前再查 |
| 航班查询字段错误 | 城市不在机场表内 | 传入 IATA 三字码，或补充 `CITY_AIRPORTS` |
| 实时步骤被跳过 | 本轮剩余时间 <25s | 查看 `plan.realtime_step_skipped` 日志；可调大 `HARNESS_DEADLINE_SECONDS` |

---

## 九、测试方式

不依赖真实 MCP 服务：

- `tests/mcp_fakes.py` 的 `FakeMCPHub` 只实现 `call/has/server_names`，按 `(server, tool)` 路由到预置响应；
- `tests/test_mcp_client.py` 用 `httpx.MockTransport` 模拟完整 SSE 流程（`endpoint` → `202` → 事件流回包 → id 配对），
  并覆盖重试、协议错误不重试、`isError`、配置解析（含参考项目 `servers_config.json` 格式）；
- `tests/test_mcp_travel_data.py` 覆盖单位换算（米/秒、公里/小时文本）、备选路线取最短、预算过滤、日期匹配、降级估算；
- `tests/test_mcp_tools.py` 覆盖草稿回填、字段级错误、`>800km` 自动查航班、`<800km` 不查、部分失败仍成功、全部失败给 `upstream_error`；
- 最后一个用例端到端跑 PLAN_EXECUTE：假 MCP + 假 LLM，断言实时工具被调用、结果写入 L2 快照。

```powershell
pytest tests -q
```

---

## 十、与参考项目的差异（为什么不是逐行移植）

| 参考项目 | 本仓库 | 原因 |
|---|---|---|
| `openai-agents` 的 `MCPServerSse` | 自实现 HTTP/SSE 客户端 | 依赖体积与版本风险；协议本身只需要 JSON-RPC + SSE |
| `print` 调试输出 | 结构化日志 `mcp.*` 事件 | 生产排障需要字段，不需要 stdout |
| 返回原始长文本 | 结构化解析 + 截断（车次 4.5k / 其它 2k 字符） | 观测值进短期记忆，必须有 token 预算 |
| 站点代码查询无缓存 | 24h TTL 缓存 | 每轮都查会白白增加一次外部调用与延迟 |
| 失败返回 `{"error": ...}` 字符串 | `ToolError(UPSTREAM_ERROR)` + hint | 模型需要知道"别重试、别编造"，而不是猜 |
| 固定调用顺序 | PLAN_EXECUTE 计划 + deadline 保护 | 外部慢查询不能拖垮方案与最终答复 |
| 航班 IATA 手写字典（含错字） | 只读表 + 后缀归一化 + 未知城市报字段错误 | 错字会静默查错机场；报错让模型可修正 |

---

## 十一、已知限制

- 外部服务可用性不在本仓库控制范围内；未配置或不可用时相关能力按上文降级，不做离线伪造；
- 城市坐标与机场表是**参考数据**，不是主数据；生产环境应替换为城市/机场主数据服务；
- 高德天气通常只提供 3~7 天预报，行程较远时只能给出"暂无预报"；
- 黄历来自第三方服务，仅作文本参考，不参与差标、报价或审批判断；
- 航班 MCP 的返回格式差异最大，当前只做文本透传（不解析班次结构），便于换供应商；
- 真实车次/航班/酒店价格仍以 12306、航司、高德与 TMC 报价为准，本仓库不提供出票能力。
