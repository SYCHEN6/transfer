# Agent 后端开发转型学习路线图

> **适用对象**：3 年 Java 开发 + 6 个月 AI Infra（Python）经验，目标：大模型 Agent 后台研发工程师
>
> **当前约束**：1 个月后开始投简历；语言优先 Java + Python，Go 暂缓；已有 Java + Spring AI + MCP + RAG 在建项目

---

## 一、差距分析总览

| JD 要求 | 你的现状 | 差距等级 |
|---|---|---|
| 高并发后端架构设计 | Java 后端 3 年，有实战经验 | 低，直接迁移 |
| Python | AI Infra 主力语言 | 低，补异步/Pydantic 细节 |
| Golang | 暂缓 | — |
| LLM Agent 运行时 | Spring AI + MCP 在建项目 | 中，需深化设计能力 |
| RAG / 知识库 | Spring AI RAG 在建项目 | 中，需补生产级细节 |
| MCP 协议 | Spring AI MCP 在建项目 | 中，已有实践基础 |
| LLM 评估体系 | 较少接触 | 高，需专项补充 |
| 大模型在线服务性能优化 | AI Infra 有底层认知 | 中，需补应用层知识 |

**核心判断**：你的在建项目已经覆盖了 JD 的主体技术，1 个月的目标不是从零学新东西，而是**把项目做深、补评估体系、准备好面试表达**。

---

## 二、1 个月冲刺计划

### 第 1 周：把在建项目做到「能讲清楚」的深度

投简历前最重要的事是有一个**能在面试中深度展开**的项目，而不是多学一门技术。

**RAG 部分需要能回答的问题：**

- 你用的什么分块策略？为什么？对比过哪几种？
- 检索召回率不够高时怎么排查？
- 有没有做混合检索（向量 + BM25）？重排序用了什么模型？
- 多租户场景下知识库怎么隔离？

**当前 Spring AI RAG 需要补的生产级细节：**

| 当前可能的状态 | 需要补充到的状态 |
|---|---|
| 固定长度分块 | 对比递归分块、父子分块，选择并说明原因 |
| 纯向量检索 | 加入 BM25 混合检索（Spring AI 支持 Elasticsearch） |
| 无重排序 | 接入 bge-reranker 或 Cohere Rerank API |
| 单租户 | 设计多租户 namespace 隔离方案（即使没实现，能说清设计） |

**MCP 部分需要能回答的问题：**

- 你实现了哪几种 MCP 原语（Tools / Resources / Prompts）？
- Tool 执行失败怎么处理？有没有超时控制？
- 如果要做 MCP 市场（多人注册工具），你会怎么设计注册/鉴权/沙箱？

**Agent Loop 部分需要能回答的问题：**

- 你的 Agent 是什么模式？ReAct 还是 Plan-and-Execute？
- 多轮对话的上下文怎么管理？有没有窗口截断？
- 工具调用结果如何回传给 LLM？并行工具调用怎么处理？

---

### 第 2 周：补 Python Agent 层 + 对齐 LangChain 体系

你已经用 Spring AI 理解了 Agent 框架，LangChain/LangGraph 的概念几乎一一对应，上手成本很低。这一周的目标是**能在 Python 里把同样的事情做一遍**，因为面试官的技术栈大概率是 Python。

**Spring AI → LangChain 对照速查：**

| Spring AI 概念 | LangChain / LangGraph 对应 |
|---|---|
| `ChatClient` / `ChatModel` | `ChatOpenAI` / `BaseChatModel` |
| `Advisor`（RAG 增强） | `RunnableWithMessageHistory` + Retriever |
| `ToolCallback` | `@tool` 装饰器 / `BaseTool` |
| `VectorStore` | `VectorStore`（接口相同） |
| Spring AI MCP Client | `langchain-mcp-adapters` |
| 无直接对应 | **LangGraph**（有状态 Agent，DAG 流程） |

**这周重点学：LangGraph**

Spring AI 目前没有对标 LangGraph 的组件。LangGraph 是面试高频考点（有状态 Agent、条件分支、循环控制），1~2 天可以跑通基础 demo：

```python
# LangGraph 最小 ReAct Agent 结构
from langgraph.graph import StateGraph, END

def should_continue(state): ...   # 决定下一步：调工具还是结束
def call_model(state): ...        # 调 LLM
def call_tools(state): ...        # 执行工具

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
```

**Python 补漏（结合项目用，不单独学）：**

- `asyncio` + `async def`：LLM 调用改异步，对比 Spring 的响应式编程
- `Pydantic v2`：工具参数 Schema 定义、LangChain 内部大量使用
- `FastAPI` + `StreamingResponse`：把你的 Agent 包一层 Python HTTP 接口

---

### 第 3 周：LLM 评估 + 性能优化知识

这两块是 JD 明确要求但你相对薄弱的，1 周够用来达到「面试能聊」的水平。

#### 评估体系（重点，JD 首条就提到）

**RAGAS 框架（RAG 评估，必须会用）：**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
```

需要理解的 4 个核心指标：

| 指标 | 评估什么 | 差时说明什么问题 |
|---|---|---|
| Context Recall | 检索到了多少相关文档 | 检索策略有问题 |
| Context Precision | 检索到的文档有多少是有用的 | 噪声太多 |
| Faithfulness | 答案是否忠于检索内容 | LLM 在幻觉 |
| Answer Relevancy | 答案是否回答了问题 | Prompt 设计有问题 |

**Agent 评估（了解思路）：**

- 任务完成率：给定目标，Agent 最终是否达成
- 工具调用准确率：选对工具、参数正确的比例
- 轨迹评估（Trajectory Eval）：对比 Agent 的推理步骤是否合理

#### LLM 在线服务性能（利用 AI Infra 背景快速消化）

你理解底层，这块消化会很快，重点在**应用层视角**：

- **TTFT / TPOT / Throughput**：这 3 个指标是面试必问，需要能说清楚优化方向
- **语义缓存**：相似问题直接返回缓存结果（向量相似度判断），是 RAG 系统降成本的常用手段
- **Prompt Cache**：OpenAI / Anthropic 的前缀缓存机制，固定的 System Prompt 不重复计费
- **降级方案**：主模型超时 → 切备用模型 → 切降级回复，需要能设计完整链路

---

### 第 4 周：系统设计准备 + 简历/表达打磨

#### 高频系统设计题（各准备一个思路框架）

**题 1：设计一个企业级 RAG 知识库系统**

```
摄入链路：文件上传 → MQ 异步处理 → 解析/分块/向量化 → 写入 VectorDB + ES
检索链路：查询 → 混合检索（向量 + BM25）→ Rerank → Top-K → LLM 生成
工程要点：多租户隔离 / 增量更新 / 检索质量监控
```

**题 2：设计一个 MCP 工具市场**

```
注册：开发者提交 MCP Server 描述 + Schema → 审核 → 上架
发现：按能力分类检索，LLM 自动匹配合适工具
执行：沙箱隔离（Docker）/ 超时控制 / 鉴权（API Key per tool）
```

**题 3：LLM 网关设计**

```
路由：按模型能力、成本、负载动态路由
限流：Token-level 限流（不只是 QPS），按用户/租户配额
降级：主模型熔断 → 备用模型 → 缓存兜底
可观测：每次调用记录 token 消耗、延迟、成本
```

#### 简历中项目的表达方式

**不好的写法（功能罗列）：**
> 使用 Spring AI + MCP + RAG 开发了一个 Agent 平台

**好的写法（有深度、有数据、有设计决策）：**
> 基于 Spring AI 设计并实现企业级 RAG Agent 平台：采用父子分块 + 混合检索（向量 + BM25）+ bge-reranker 重排序方案，检索召回率提升 X%；实现 MCP Server 支持工具动态注册，工具执行引入超时熔断机制；设计多租户知识库隔离方案，支持 namespace 级权限控制

> （没有真实数据时，把设计决策和解决了什么问题说清楚，比堆技术名词有效）

---

## 三、知识备查（面试前速查，不需要提前全学）

### RAG 核心知识点

**分块策略对比：**

| 策略 | 适用场景 | 缺点 |
|---|---|---|
| 固定长度 | 快速上线，结构化文档 | 切断语义完整性 |
| 递归分块 | 通用场景首选 | 参数需调优 |
| 父子分块 | 需要上下文时 | 实现复杂度高 |
| 语义分块 | 质量要求高 | 计算成本高 |

**向量数据库选型：**

| 数据库 | 特点 | 推荐场景 |
|---|---|---|
| Milvus | 生产级，功能最全 | 企业部署首选 |
| Qdrant | 轻量，Rust 实现 | 中小规模 |
| pgvector | PostgreSQL 插件 | 已有 PG 的场景 |
| Elasticsearch | 混合检索原生支持 | 需要全文搜索 |

### Agent 设计模式速查

| 模式 | 适用场景 | Spring AI 支持 |
|---|---|---|
| ReAct | 通用工具调用 Agent | 是（`ChatClient` + Tools） |
| Plan-and-Execute | 复杂多步骤任务 | 需自行实现 |
| Multi-Agent | 任务分工、并行执行 | 部分支持 |
| Human-in-the-Loop | 需要人工审批 | 需自行实现 |

### LLM 服务性能指标速查

| 指标 | 含义 | 典型优化手段 |
|---|---|---|
| TTFT（首 Token 延迟） | 从请求到收到第一个 Token | Prompt Cache、减少 Prefill 长度 |
| TPOT（每 Token 时间） | 生成速度 | 模型量化、批处理优化 |
| Throughput（吞吐量） | 单位时间 token 数 | Continuous Batching |
| P99 延迟 | 尾延迟，影响 SLA | 限流、超时控制、熔断 |

---

## 四、推荐资源（按优先级排序）

**第 1 周用：**
- Spring AI 官方文档 → RAG / MCP 章节（对照你的项目补细节）
- RAGAS 官方文档（`docs.ragas.io`）

**第 2 周用：**
- LangGraph 官方教程（`langchain-ai.github.io/langgraph`）→ Quickstart + Concepts 两篇够了
- LangChain 与 Spring AI 对照表（自己整理一份，面试时能说清楚迁移思路）

**第 3 周用：**
- vLLM 官方博客（PagedAttention 论文摘要版）→ 理解 KV Cache 原理
- Langfuse 文档 → 理解 LLM 可观测性设计

**开源项目（读架构，不需要读完）：**
- **Dify**：完整 RAG + Agent 平台，Python，能帮你看清生产级系统的全貌
- **RAGFlow**：专注 RAG 质量，文档解析做得很深

---

## 五、核心竞争力定位

**优势 1：AI Infra 背景（面试时主动展开）**

> "我在 AI Infra 阶段做过华为昇腾 NPU 上的推理适配，理解 KV Cache、Attention 计算的底层机制。这让我在做 LLM 在线服务性能优化时，能从推理框架层定位瓶颈，不只停留在应用层加缓存。"

**优势 2：Spring AI + MCP 实战项目（当前正在建）**

MCP 是 2024 年底才发布的新标准，大多数候选人只是听说过，你已经在项目里实现了，是真实的差异化。

**优势 3：Java 高并发工程经验**

设计大规模 Agent 系统（消息队列、分布式锁、多租户隔离）时，Java 3 年的工程积累直接迁移。

**关于语言（面试时怎么回应 Go 问题）：**

如果面试官问 Go，直接说：「目前主要用 Java 和 Python，Go 可以读懂代码，上手时间预计 2~3 周。」不要回避，展示学习能力比假装会更有效。

---

---

## 六、重点学习内容与资源

> 以下三块是边做项目边必须打牢的基础。每块分「必须掌握的具体内容」和「按顺序学的资源」两部分。

---

### 6.1 RAG

#### 必须掌握的内容

**① 文档分块（Chunking）**

- 固定长度分块：最简单，但会切断句子和段落，适合格式规整的短文档
- 递归字符分块（RecursiveCharacterTextSplitter）：按段落→句子→字符逐级退化，通用场景首选
- 父子分块（Parent-Child Chunking）：小块用于精准检索，大块作为上下文返回给 LLM，解决「检索到了但上下文不完整」的问题
- 语义分块：用 embedding 相似度决定分块边界，质量最高但计算成本也最高
- **核心判断依据**：文档是否有天然结构（标题/段落）？检索时需要多少上下文？对延迟的要求？

**② 检索策略**

- 稠密检索（Dense Retrieval）：将查询和文档都转成向量，用余弦相似度或点积排序
- 稀疏检索（BM25）：基于关键词匹配，对专有名词、代码、型号等精确词检索效果远好于向量
- 混合检索（Hybrid Search）：两路结果用 RRF（Reciprocal Rank Fusion）算法合并排名，生产环境标配
- 重排序（Rerank）：召回 Top-20，再用 cross-encoder 模型精排取 Top-5，显著提升精度
- **必须理解**：为什么单纯向量检索在专业领域效果差？RRF 公式是什么？

**③ Embedding 模型**

- 多语言中文场景：BGE 系列（`BAAI/bge-large-zh-v1.5`）或 `bce-embedding-base_v1`
- 通用场景：OpenAI `text-embedding-3-small`（性价比高）
- **必须理解**：embedding 维度对检索性能和存储的影响；同模型 embed 查询和文档的重要性

**④ 向量数据库核心概念**

- ANN 索引类型：HNSW（图索引，召回率高，内存占用大）vs IVF（倒排索引，内存友好）
- 元数据过滤：先按 metadata 条件筛选，再做向量检索（多租户隔离的实现方式）
- Namespace / Collection 隔离：不同用户/租户的数据物理隔离

**⑤ 生成质量控制**

- Prompt 模板设计：如何指示 LLM「只用检索到的内容回答，不要编造」
- 上下文长度控制：召回文档过多时如何截断，避免超过 context window
- 引用溯源：答案中标注来源文档，提升可信度

#### 学习资源（按顺序）

| 顺序 | 资源 | 说明 |
|---|---|---|
| 1 | [Spring AI 官方文档 RAG 章节](https://docs.spring.io/spring-ai/reference/) | 对照你的项目，理解框架封装了什么 |
| 2 | [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/) | Python 视角，补混合检索和重排序实现 |
| 3 | [DeepLearning.AI《Building and Evaluating Advanced RAG》](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) | 免费短课，1~2 小时，父子分块和评估讲得最清楚 |
| 4 | [RAGAS 官方文档](https://docs.ragas.io) | 学会用 4 个核心指标评估你的 RAG 效果 |
| 5 | [Milvus 文档混合检索章节](https://milvus.io/docs/) | 理解 RRF 算法和生产级索引配置 |

---

### 6.2 MCP（Model Context Protocol）

#### 必须掌握的内容

**① 协议核心概念**

- **三种原语的区别**：
  - `Tools`：LLM 可以主动调用的函数，有输入参数和返回值（如「查天气」「执行代码」）
  - `Resources`：LLM 可以读取的数据源，URI 寻址（如「读取文件」「查询数据库」）
  - `Prompts`：预定义的提示模板，用户可触发（如「代码审查模板」）
- **Client-Server 通信流程**：
  1. Client 连接 Server，发起 capability negotiation
  2. Server 返回其支持的 Tools/Resources/Prompts 列表（及 Schema）
  3. LLM 根据 Schema 决定调用哪个工具，生成符合 Schema 的参数
  4. Client 将调用请求发给 Server 执行，返回结果
- **传输层**：Stdio（本地进程）和 HTTP+SSE（远程服务）两种方式

**② Server 实现要点**

- Tool Schema 设计：`description` 写好很关键，LLM 根据 description 决定什么时候调用这个工具
- 参数验证：收到 LLM 生成的参数后必须做校验，LLM 可能生成不合规的参数
- 错误返回：Tool 执行失败时返回结构化错误，让 LLM 能理解并决定下一步
- 超时控制：每个 Tool 调用必须有超时，防止 Agent 卡死

**③ 安全与工程设计（面试加分项）**

- Tool 执行沙箱：代码执行类 Tool 必须隔离（Docker / 进程隔离），防止恶意代码
- 权限控制：不同用户能调用的 Tool 不同，在 Client 层做鉴权
- MCP 市场架构：Tool 注册（Schema 审核）→ 发现（分类检索）→ 动态加载到 Agent

#### 学习资源（按顺序）

| 顺序 | 资源 | 说明 |
|---|---|---|
| 1 | [MCP 官方介绍](https://modelcontextprotocol.io/introduction) | 30 分钟读完，建立整体概念 |
| 2 | [MCP 协议规范](https://spec.modelcontextprotocol.io) | 重点看 Tools 和通信流程两章，不需要全读 |
| 3 | [Spring AI MCP 文档](https://docs.spring.io/spring-ai/reference/api/mcp/) | 对照你的项目，理解 Java SDK 封装层 |
| 4 | [MCP Java SDK 源码](https://github.com/modelcontextprotocol/java-sdk) | 看 examples 目录，理解 Tool 注册和调用流程 |
| 5 | [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | 用 Python 再实现一个 MCP Server，加深理解 |

---

### 6.3 Agent

#### 必须掌握的内容

**① ReAct 循环机制（最核心）**

必须能在不依赖框架的情况下手写一个最小 ReAct Loop：

```
思考（Thought）→ 行动（Action，即工具调用）→ 观察（Observation，工具结果）→ 循环
```

关键细节：
- 每轮循环，完整的消息历史（含之前所有 Thought/Action/Observation）都要传给 LLM
- LLM 生成「Final Answer」时循环结束
- 工具调用结果以 `tool` role 消息追加到历史中

**② 上下文与记忆管理**

- **短期记忆（对话历史）**：如何做滑动窗口截断，避免超出 context window；截断时保留哪些消息
- **长期记忆**：重要信息 embed 后存向量库，下次对话时检索注入；或用摘要压缩旧历史
- **会话持久化**：多轮对话的状态存 Redis 或数据库，服务重启不丢失

**③ 工具调用工程细节**

- **并行工具调用**：LLM 一次返回多个 tool_call，需并发执行后统一返回结果
- **工具调用失败处理**：返回错误描述给 LLM，让其决定重试还是换策略
- **防止无限循环**：设置最大迭代次数（max_iterations），超出则强制返回当前状态
- **工具调用幂等**：同一工具同参数重复调用结果应一致，或做去重处理

**④ 主流 Agent 设计模式（了解，面试能说清楚）**

| 模式 | 核心思路 | 适用场景 |
|---|---|---|
| ReAct | 推理和行动交替，单 Agent | 通用工具调用，大多数场景 |
| Plan-and-Execute | 先整体规划，再逐步执行 | 复杂多步骤任务，步骤间依赖强 |
| Multi-Agent | Orchestrator 分解任务给多个专用 Agent | 任务可并行、需要专业分工 |
| Reflection | 执行完后自我评估并修正 | 对质量要求高，允许多次尝试 |

**⑤ LangGraph（Python Agent 框架，面试高频）**

Spring AI 没有对标 LangGraph 的组件。LangGraph 用图结构表达 Agent 流程，比链式结构更灵活：

- **Node**：一个处理步骤（调 LLM / 执行工具 / 人工审批）
- **Edge**：步骤间的跳转，可以是条件跳转（根据 LLM 输出决定下一步）
- **State**：贯穿整个图的共享状态，每个节点读写
- **Checkpointing**：自动保存每步状态到数据库，支持暂停/恢复（Human-in-the-Loop 的基础）

#### 学习资源（按顺序）

| 顺序 | 资源 | 说明 |
|---|---|---|
| 1 | [Anthropic《Building Effective Agents》](https://www.anthropic.com/research/building-effective-agents) | 最值得读的 Agent 工程文章，讲清楚了何时用 Agent、常见设计模式 |
| 2 | [LangGraph 官方教程](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | Quickstart 跑通后读 Concepts 章节，理解 State/Node/Edge |
| 3 | [DeepLearning.AI《AI Agents in LangGraph》](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | 免费短课，4 小时，从 ReAct 手写到 LangGraph 实现，进阶讲 Human-in-the-Loop |
| 4 | [ReAct 原始论文](https://arxiv.org/abs/2210.03629) | 读 Abstract + Introduction 即可，理解设计动机 |
| 5 | [Spring AI Advisors 文档](https://docs.spring.io/spring-ai/reference/api/advisors.html) | 理解 Spring AI 的 Agent 抽象和 LangChain 的差异 |

---

### 6.4 LLM 评估（差距最大的一块）

#### 必须掌握的内容

**① RAG 评估四指标（用 RAGAS 框架）**

| 指标 | 评估什么 | 差时说明什么 |
|---|---|---|
| Context Recall | 相关文档有没有被召回 | 检索策略有问题，重点优化分块和检索 |
| Context Precision | 召回的文档有多少是有用的 | 噪声太多，考虑加重排序 |
| Faithfulness | 答案是否忠于检索内容，没有编造 | LLM 在幻觉，优化 Prompt 或换模型 |
| Answer Relevancy | 答案是否回答了问题 | Prompt 设计问题或检索质量差 |

**② 评估数据集构建**

- Golden Dataset：准备 50~100 条「问题 + 标准答案 + 相关文档」的测试集
- 自动生成测试集：用 LLM 根据文档生成问题（RAGAS 内置此功能）
- 版本管理：每次优化后跑评估，对比指标变化，判断优化是否有效

**③ Agent 评估思路（了解）**

- 任务完成率：给定最终目标，Agent 是否达成（人工标注或 LLM 裁判）
- 工具调用准确率：调用了正确的工具、传入了正确的参数
- 轨迹合理性：推理步骤是否高效，有没有不必要的工具调用

#### 学习资源

| 顺序 | 资源 | 说明 |
|---|---|---|
| 1 | [RAGAS 官方文档](https://docs.ragas.io/en/latest/getstarted/) | 先跑通 Quickstart，再看各指标的计算原理 |
| 2 | [DeepLearning.AI《Building and Evaluating Advanced RAG》](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) | 课程后半段专门讲评估，和 RAGAS 配合使用 |
| 3 | [DeepEval 文档](https://docs.confident-ai.com) | 更通用的评估框架，支持 Agent 评估，可以补充了解 |

---

---

## 七、练手项目：代码 Review Agent

> **目标**：输入 GitHub PR 链接，Agent 自动拉取变更、检索编码规范、输出结构化 review 意见
>
> **预计周期**：2~3 周，可利用碎片时间推进
>
> **语言**：Python（补 LangGraph）；MCP Server 可用 Java 或 Python 二选一

---

### 7.1 前置准备

#### 环境依赖

```bash
# 包管理（用 uv，比 pip 快很多）
pip install uv
uv venv && source .venv/bin/activate

# 核心依赖
uv add langgraph langchain-openai langchain-community
uv add mcp                          # MCP Python SDK
uv add fastapi uvicorn              # HTTP 接口层
uv add qdrant-client                # 向量数据库（本地运行，无需部署）
uv add ragas                        # 评估框架
uv add PyGithub                     # GitHub API 客户端
```

#### 需要申请的 API Key / Token

| 依赖 | 用途 | 获取方式 |
|---|---|---|
| OpenAI API Key | LLM 推理 + Embedding | `platform.openai.com`，也可替换为国内厂商 |
| GitHub Personal Token | 调用 GitHub API 拉取 PR | GitHub → Settings → Developer settings → Tokens |

#### 编码规范知识库文档（RAG 的数据源）

准备 2~3 份编码规范文档作为知识库，推荐：
- [阿里巴巴 Java 开发手册](https://github.com/alibaba/p3c)（PDF 版，直接下载）
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)（可用 `wget` 抓取 HTML）
- 自定义规范：用 Markdown 写几条你自己项目的约定（真实感更强，面试时好展示）

---

### 7.2 整体架构

```
用户输入 PR URL
       │
       ▼
┌─────────────────────────────────────────┐
│              LangGraph Agent            │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │ fetch_pr │───▶│  analyze_files   │   │
│  │  (MCP)   │    │  (并行，每文件)   │   │
│  └──────────┘    └────────┬─────────┘   │
│                           │             │
│                  ┌────────▼─────────┐   │
│                  │ retrieve_rules   │   │
│                  │  (RAG 检索规范)  │   │
│                  └────────┬─────────┘   │
│                           │             │
│                  ┌────────▼─────────┐   │
│                  │ generate_review  │   │
│                  │  (LLM 综合输出)  │   │
│                  └──────────────────┘   │
└─────────────────────────────────────────┘
       │
       ▼
结构化 Review 报告（问题列表 + 等级 + 建议）
```

**LangGraph State 设计：**

```python
from typing import TypedDict, List

class ReviewState(TypedDict):
    pr_url: str
    pr_diff: str                  # fetch_pr 节点填充
    changed_files: List[str]      # fetch_pr 节点填充
    file_analyses: List[dict]     # analyze_files 节点填充
    relevant_rules: List[str]     # retrieve_rules 节点填充
    review_comments: List[dict]   # generate_review 节点填充
```

---

### 7.3 分步实现

#### Step 1：MCP Server — GitHub 工具（2~3 天）

实现 3 个 Tool，覆盖 PR 数据获取的完整需求：

```python
# github_mcp_server.py
from mcp.server import FastMCP
from github import Github

mcp = FastMCP("github-review-tools")

@mcp.tool()
def get_pr_diff(repo: str, pr_number: int) -> str:
    """获取 PR 的完整 diff 内容"""
    ...

@mcp.tool()
def list_pr_files(repo: str, pr_number: int) -> list[dict]:
    """列出 PR 变更的文件列表及变更类型（added/modified/deleted）"""
    ...

@mcp.tool()
def get_file_content(repo: str, file_path: str, ref: str) -> str:
    """获取指定 commit 的文件完整内容（用于理解上下文）"""
    ...
```

**需要注意的问题**：
- PR diff 可能非常长（数千行）→ 按文件分割，每个文件单独处理
- 二进制文件、锁文件（`package-lock.json` 等）直接跳过，不送给 LLM

#### Step 2：RAG 知识库搭建（2~3 天）

```python
# build_knowledge_base.py
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

# 1. 加载文档
docs = []
docs += PyPDFLoader("alibaba-java-handbook.pdf").load()
docs += WebBaseLoader("https://google.github.io/styleguide/pyguide.html").load()

# 2. 分块（编码规范适合按条目分块，chunk_size 不需要太大）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量化 + 存储
vectorstore = Qdrant.from_documents(
    chunks,
    OpenAIEmbeddings(),
    location=":memory:",   # 本地内存，demo 够用
    collection_name="coding_rules"
)
```

**需要验证的点**：写几条测试查询（如「如何命名变量」「空指针异常处理」），人工检查召回的规范条目是否相关，不相关再调整分块策略。

#### Step 3：LangGraph Agent 主体（3~4 天）

```python
# agent.py
from langgraph.graph import StateGraph, END

def fetch_pr_node(state: ReviewState) -> ReviewState:
    """调用 MCP Tool 拉取 PR 数据"""
    # 解析 PR URL，调用 get_pr_diff + list_pr_files
    ...

def analyze_files_node(state: ReviewState) -> ReviewState:
    """对每个变更文件提取关键信息（变更类型、主要改动）"""
    # 可并行处理多文件：asyncio.gather
    ...

def retrieve_rules_node(state: ReviewState) -> ReviewState:
    """根据文件类型 + 变更内容，检索相关编码规范"""
    # 查询向量库，返回 Top-5 相关规范条目
    ...

def generate_review_node(state: ReviewState) -> ReviewState:
    """综合 diff + 规范，生成结构化 review"""
    # 输出格式：[{file, line, severity, issue, suggestion}]
    ...

# 构建图
graph = StateGraph(ReviewState)
graph.add_node("fetch_pr", fetch_pr_node)
graph.add_node("analyze_files", analyze_files_node)
graph.add_node("retrieve_rules", retrieve_rules_node)
graph.add_node("generate_review", generate_review_node)

graph.set_entry_point("fetch_pr")
graph.add_edge("fetch_pr", "analyze_files")
graph.add_edge("analyze_files", "retrieve_rules")
graph.add_edge("retrieve_rules", "generate_review")
graph.add_edge("generate_review", END)

app = graph.compile()
```

**需要注意的问题**：
- `generate_review` 的 Prompt 要明确要求 LLM「只基于提供的规范条目给出意见，不要编造规则」—— 控制幻觉
- 输出格式用 `with_structured_output(ReviewComment)` 强制 JSON，方便后续处理

#### Step 4：FastAPI 接口层（1 天）

```python
# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

api = FastAPI()

@api.post("/review")
async def review_pr(pr_url: str):
    """同步版本，返回完整 review"""
    result = app.invoke({"pr_url": pr_url})
    return result["review_comments"]

@api.post("/review/stream")
async def review_pr_stream(pr_url: str):
    """流式版本，实时返回每个节点进度"""
    async def generate():
        async for event in app.astream({"pr_url": pr_url}):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

> 流式接口要做：既是练 SSE 实现，也是面试展示亮点（「用户能实时看到 Agent 在做什么」）。

#### Step 5：RAGAS 评估（1~2 天）

准备 10~15 条测试样本（手工构造即可，不需要多）：

```python
# 每条样本的结构
{
    "pr_url": "github.com/xxx/yyy/pull/123",
    "question": "这个 PR 有哪些编码规范问题？",
    "ground_truth": "1. 变量命名不符合驼峰规范（第 23 行）2. 未处理空指针异常（第 45 行）",
    "contexts": ["召回的规范条目..."]   # 由 RAG 自动填充
}
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
print(results)
# 根据结果决定优化方向：分块？检索？Prompt？
```

---

### 7.4 预期遇到的问题和解法

| 问题 | 原因 | 解法 |
|---|---|---|
| PR diff 超出 context window | 大型 PR 动辄数千行 | 按文件分批处理，每次只送单文件 diff |
| Review 意见太泛，没有具体行号 | LLM 没有足够上下文 | Prompt 明确要求「指出文件名和行号」，同时传入文件完整内容而非只传 diff |
| RAG 召回规范不相关 | 分块粒度太大，或查询语义偏差 | 改用「问题 + 代码片段」作为查询，缩小 chunk_size |
| RAGAS 评估 Faithfulness 低 | LLM 在编造规则 | Prompt 加约束：「严格基于 <context> 中的规范，不得引用未提供的规则」 |
| GitHub API 限流 | 未认证调用限制 60次/小时 | 使用 Personal Token，限制提升到 5000次/小时 |

---

### 7.5 可扩展的进阶功能（时间够再做）

- **支持多语言**：根据文件扩展名自动选择对应语言的规范知识库
- **Human-in-the-Loop**：LangGraph Checkpointing，生成 review 前暂停让用户确认规范范围
- **历史 review 缓存**：同一 PR 二次请求直接返回缓存结果（语义缓存练习）
- **指标看板**：统计每次 review 的 Token 消耗、延迟、RAGAS 分数，用 Langfuse 记录

---

### 7.6 简历上怎么写这个项目

> 基于 LangGraph + MCP + RAG 设计并实现代码 Review Agent：通过 MCP GitHub 工具拉取 PR diff，结合编码规范知识库（RAG 混合检索 + bge-reranker 重排序）生成结构化 review 意见；实现 SSE 流式接口实时推送 Agent 推理过程；使用 RAGAS 框架构建评估 Pipeline，Faithfulness 指标达到 X，Context Recall 达到 X。

---

*文档更新日期：2026-04-13*
