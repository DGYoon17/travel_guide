# 压测手册

压测要分三层，不能混在一起：

| 层 | 压什么 | 是否需要真实模型 | 能回答的问题 |
|---|---|---|---|
| A. 编排层 | 路由、意图、工具、记忆、会话 | 否（固定时延假模型） | 服务端自身是否稳、并发是否线性、路由少调几次模型 |
| B. HTTP/SSE | FastAPI、SSE、中间件、序列化 | 否（OpenAI 兼容 mock） | 接口层吞吐、TTFT、连接/流式开销 |
| C. 真实模型 | 端到端质量与时延 | 是 | 真实 TTFT、token 成本、限流、上下文影响 |

**只有 C 层的结果能直接写进简历。** A/B 层用于定位瓶颈、验证改造方向，不能冒充线上模型指标。

---

## 一、A 层：编排层压测（已跑通）

### 命令

```powershell
python benchmarks/bench_inproc.py `
  --concurrency 1 10 30 `
  --requests 15 `
  --delay-ms 50
```

参数：

- `--concurrency`：同时运行的 worker 数；
- `--requests`：每个 worker 的请求数；
- `--delay-ms`：假模型每次调用的固定时延。

脚本会给每个 worker 独立的 `InMemoryTravelSessionStore` 和 `TravelOrchestrator`，
只测 `orchestrator.run()`，不含 HTTP、网络、真实模型。

### 覆盖场景

| 场景 | 用户输入 | 路由 on | 路由 off（基线） |
|---|---|---|---|
| `policy` | 公司差旅住宿标准是多少 | DIRECT，1 次模型调用 | REACT，2 次 |
| `plan` | 帮我安排一下行程 | PLAN_EXECUTE，1 次 | REACT，2 次 |
| `chat` | 你好 | DIRECT，1 次 | DIRECT，1 次 |
| `booking_gate` | 帮我订这个方案 | REACT + 确认门，1 次 | 同左 |

### 已跑结果（`delay-ms=50`，每 worker 15 请求）

| 场景 | 模式 | 并发 | P50 | P95 | 模型调用/请求 |
|---|---|---:|---:|---:|---:|
| policy | router on | 1 | 66.4ms | 218.5ms | 1.00 |
| policy | force react | 1 | 133.1ms | 174.6ms | 2.00 |
| policy | router on | 30 | 83.9ms | 96.6ms | 1.00 |
| policy | force react | 30 | 278.2ms | 408.6ms | 2.00 |
| plan | router on | 30 | 114.1ms | 128.2ms | 1.00 |
| plan | force react | 30 | 306.5ms | 424.9ms | 2.00 |

结论：**在模型单次时延固定的前提下，端到端时延近似随"模型调用次数"线性变化**。
`policy` 从 2 次降到 1 次，P50 下降约 50%~70%（并发越高，队列效应越明显）。

注意：这是假模型，不能证明真实模型下的 30%。

### 结果文件

`benchmarks/reports/inproc_*.json`

---

## 二、B 层：HTTP/SSE 压测（已跑通）

### 1. 启动 mock 模型服务

```powershell
$env:MOCK_LLM_DELAY_MS="20"
python -m uvicorn benchmarks.mock_openai_server:app --host 127.0.0.1 --port 18081
```

它实现 OpenAI 兼容的：

- `/v1/chat/completions`（流式 content / tool_calls，非流式分类/摘要）
- `/v1/models`、`/health`

### 2. 启动应用（路由 on）

```powershell
$env:DASHSCOPE_API_BASE="http://127.0.0.1:18081/v1"
$env:DASHSCOPE_API_KEY="dummy"
$env:AUTH_REQUIRED="true"
$env:AUTH_DEV_TOKEN="bench-token"
$env:REDIS_URL="redis://127.0.0.1:1/0"          # 故意不可用，走内存降级
$env:DATABASE_URL="postgresql+asyncpg://u:p@127.0.0.1:1/db"
$env:MILVUS_HOST="127.0.0.1"
$env:MILVUS_PORT="19531"
$env:ROUTER_MODE="on"
python -m uvicorn app.main:app --host 127.0.0.1 --port 18080
```

基线用同样命令，只把 `ROUTER_MODE` 改成 `off`，端口换 18082。

> 生产压测不要用 `127.0.0.1:1` 这种不可用依赖，应该起真实 Redis/PostgreSQL/Milvus，
> 否则测到的是"降级路径"。

### 3. 发压

```powershell
python benchmarks/load_http.py `
  --url http://127.0.0.1:18080 `
  --token bench-token `
  --concurrency 1 5 10 `
  --requests 10 `
  --message "公司差旅住宿标准是多少"
```

`--reuse-session` 可以让同一 worker 复用会话，用来压长期记忆增长与摘要触发。

### 4. 已跑结果（mock 每次调用 20ms）

| 模式 | 并发 | RPS | TTFT P50 | TTFT P95 | 总时长 P50 | 总时长 P95 | 成功 |
|---|---:|---:|---:|---:|---:|---:|---:|
| router on | 1 | 20.8 | 31.1ms | 44.9ms | 34.0ms | 47.6ms | 10/10 |
| router off | 1 | 6.8 | 80.4ms | 628.0ms | 83.3ms | 631.0ms | 10/10 |
| router on | 5 | 75.3 | 40.8ms | 105.0ms | 51.0ms | 117.4ms | 50/50 |
| router off | 5 | 38.7 | 97.5ms | 206.3ms | 108.3ms | 215.6ms | 50/50 |
| router on | 10 | 89.3 | 74.4ms | 156.3ms | 94.5ms | 177.1ms | 100/100 |
| router off | 10 | 49.8 | 130.6ms | 228.9ms | 149.2ms | 249.0ms | 100/100 |

结论：

- P50 总时长从 83ms 降到 34ms（并发 1，约 −59%），从 149ms 降到 94.5ms（并发 10，约 −37%）；
- RPS 约 1.8~3 倍；
- 差异来源不是"代码变快"，而是 **baseline 每个请求多一次模型调用**；
- 这是 mock 模型下的**机制验证**，绝对值不能当线上时延。

### 5. 结果文件

`benchmarks/reports/http_*.json`

---

## 三、C 层：真实模型小并发实验

真实模型不要压高并发：成本高、容易被限流，而且测出来是供应商的队列，不是你的服务。

建议：

- 并发 1 / 3 / 5，每档 20 个请求；
- 固定模型版本、`temperature=0`、固定 system prompt 版本；
- 用同一个数据集（`evals/dataset/travel_cases.jsonl`）；
- 记录 TTFT、总时长、prompt/completion token、每次请求的模型调用次数；
- 同时记录 `ROUTER_MODE=on/off` 两组。

```powershell
# 真实 key
$env:DASHSCOPE_API_KEY="sk-..."
$env:DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

python benchmarks/load_http.py `
  --url http://127.0.0.1:8000 `
  --token $env:AUTH_DEV_TOKEN `
  --concurrency 1 3 5 `
  --requests 20 `
  --message "公司差旅住宿标准是多少"
```

要测不同场景，把 `--message` 换成：

- 规划：`帮我安排下周去北京出差`
- 预订：`帮我订这个方案`（需要先通过 `/chat` 生成方案并拿到 `action_id`，否则只会到确认门）
- 差标：`公司差旅住宿标准是多少`

### 关键：token 与成本

`/metrics` 里有 `tokens_prompt` / `tokens_completion`，每次实验前后各取一次差值。
真实模型实验必须报告 token/请求，否则"时延下降"可能只是回答变短了。

---

## 四、指标定义

| 指标 | 定义 | 采集点 |
|---|---|---|
| TTFT | 请求发出到第一个 `content` 事件 | `load_http.py` |
| 总时长 | 请求发出到 `done` 事件 | `load_http.py` |
| RPS | 完成请求数 / 墙钟 | 脚本 |
| P50/P95 | 分位数，不看平均值 | 脚本 |
| 模型调用/请求 | `stream_turn` 次数 | `bench_inproc.py` / metrics |
| 无效工具调用率 | 参数错/缺参/前置未满足/未知工具 / 总工具调用 | `evals/run_eval.py` |
| 任务成功率 | 期望工具被调用且 finish_reason 符合 | `evals/run_eval.py` |

**不要只报平均值。** 压测里 P95 才是用户体感；平均值会被大量快请求稀释。

---

## 五、压测时应该改哪些变量

| 变量 | 怎么改 | 看什么 |
|---|---|---|
| 路由 | `ROUTER_MODE=on/off` | 模型调用次数、P50/P95 |
| 并发 | `--concurrency 1 5 10 30` | RPS 是否线性、P95 拐点 |
| 会话 | `--reuse-session` | 记忆增长、摘要触发后的时延 |
| 模型时延 | `MOCK_LLM_DELAY_MS` | 服务端开销占比 |
| Redis | 真 Redis vs 不可用 | CAS 冲突率、会话读写 P95 |
| 工具 | 只读工具并发 vs 顺序 | 工具阶段耗时 |
| 预算 | `HARNESS_MAX_*` | 触顶率、收尾质量 |

### 建议的瓶颈定位顺序

1. 先看 `/metrics` 的 `llm_stream` P50 与 `tool_*` P50，确认时间花在哪；
2. 再看 Redis/DB 连接池与超时；
3. 最后才怀疑 Python 编排层——它通常不是瓶颈。

---

## 六、纪律

1. **Mock 结果只用于机制验证**，简历写"压测下降 X%"必须来自真实模型 + 可复现报告；
2. 两组实验必须同数据集、同模型版本、同 `temperature`；
3. 报告里同时给 P50、P95、错误率、token/请求；
4. 没有报告的数字不要写；
5. 压测前先确认依赖是真实实例，不要用"端口 1"的降级路径冒充生产。

---

## 七、清理

```powershell
# 停掉实验进程（按实际 PID/端口）
Get-NetTCPConnection -LocalPort 18080,18081,18082 -ErrorAction SilentlyContinue |
  Select-Object LocalPort, OwningProcess
```

如果是用本仓库的后台任务启动的，直接结束对应 job 或进程即可。
