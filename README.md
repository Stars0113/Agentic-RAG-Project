# 企业级 Agentic RAG 医疗知识库问答系统

基于主流 RAG 架构设计，实现 **混合检索 + 三级渐进式过滤 + Agent 自主决策重试 + 语义缓存**，解决传统 RAG 系统"检索噪音大、无结果硬答、重复计算成本高"的痛点。

## 核心技术亮点

1. **混合检索 + RRF 融合 + Cross-Encoder 重排序**
   - 并行执行 BM25（关键词精准匹配）与 Vector（语义泛化匹配），通过 RRF 算法融合双路结果，再接入 `bge-reranker-large` 交叉编码器精排，确保 Top-K 结果高相关性。
2. **企业级 L0-L2 三级渐进式过滤**
   - **L0 规则过滤**：剔除黑名单词（广告/乱码）及超长超短无效文本。
   - **L1 向量过滤**：基于余弦相似度划分"强相关"（>0.60）与"边缘相关"（0.35~0.60），剔除语义偏差文档。
   - **L2 大模型 NLI 评估**：利用 LLM 判断文档是否真正具备回答问题的能力（置信度>0.6），解决"语义相似但答非所问"的问题。
3. **Agent 自主决策与查询改写**
   - 检索结果不足时，Agent 拒绝直接回答，自主判断 `LACK` 或 `SERIOUS_LACK`。
   - 自动调用 LLM 改写 Query 或补充领域关键词，最多重试 3 次直至信息充足（`ENOUGH`）。
4. **Redis 语义缓存**
   - 基于向量相似度匹配历史问答，相似度阈值>0.75 直接命中缓存，毫秒级响应。
5. **Ragas 自动化评估体系**
   - 内置 `Faithfulness`（忠实度）与 `Context Recall`（上下文召回率）量化评估，生成带等级的诊断报告。

## 系统架构

```
用户查询
  │
  ▼
┌─────────────┐     ┌──────────────────────────────────────────┐
│ Redis 语义缓存│────▶│ 命中缓存 → 直接返回结果（毫秒级）           │
└─────────────┘     └──────────────────────────────────────────┘
  │ 未命中
  ▼
┌─────────────────────────────────────────────────────┐
│                   混合检索阶段                         │
│  ┌──────────────┐       ┌──────────────────┐         │
│  │ BM25 关键词召回│       │ Vector 向量语义召回│         │
│  │ (Elasticsearch)│      │ (bge-large-zh)   │         │
│  └──────┬───────┘       └────────┬─────────┘         │
│         └────────┬───────────────┘                    │
│                  ▼                                    │
│         RRF 算法融合排名                               │
│                  ▼                                    │
│  bge-reranker-large Cross-Encoder 精排                │
└─────────────────────────┬───────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────┐
│                 三级渐进式过滤                          │
│  L0 规则过滤（黑名单/长度）                             │
│      ▼                                               │
│  L1 向量过滤（强相关 / 边缘相关）                       │
│      ▼                                               │
│  L2 LLM NLI 评估（仅对边缘文档）                        │
└─────────────────────────┬───────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────┐
│                 Agent 决策阶段                         │
│  文档充足(≥5条) → ENOUGH → 生成回答                    │
│  文档不足(2~4条) → LACK → 改写查询重试                  │
│  文档极少(<2条) → SERIOUS_LACK → 改写查询重试           │
└─────────────────────────────────────────────────────┘
```

## 项目结构

```
Agentic-RAG-Project/
├── core/                           # 核心业务逻辑
│   ├── es_client.py                # ES 连接与文档向量化入库
│   ├── llm_client.py               # 千问大模型客户端
│   ├── retriever.py                # 混合检索器编排
│   └── agent.py                    # Agent 决策与查询改写
├── config/                         # 集中化配置管理
│   ├── es_config.py                # Elasticsearch 配置
│   ├── model_config.py             # 向量/重排/LLM 模型及阈值配置
│   ├── path_config.py              # 文件路径配置
│   └── redis_config.py             # Redis 缓存配置
├── utils/                          # 通用工具集
│   ├── rank_utils.py               # RRF 融合与重排序算法
│   ├── filter_utils.py             # L0/L1/L2 三级过滤核心实现
│   ├── vector_utils.py             # 向量相似度计算
│   ├── eval_utils.py               # Ragas 评估与报告生成
│   └── cache_utils.py              # Redis 语义缓存读写
├── pipeline/                       # 流程编排
│   └── rag_pipeline.py             # 完整 RAG 流程统一编排入口
├── docs/                           # 知识库文档
│   └── medical_kb_clean.txt        # 医疗知识库（358条）
├── data/                           # 运行时输出
│   ├── bm25_results.txt            # BM25 召回结果
│   ├── vec_results.txt             # 向量召回结果
│   ├── rrf_results.txt             # RRF 融合结果
│   └── final_results.txt           # 最终检索结果
├── tests/                          # 单元测试
│   └── test_rag.py                 # 核心模块测试用例
├── main.py                         # 程序入口
├── test_retrieval.py               # 检索质量诊断脚本
├── fix_kb.py                       # 知识库格式转换工具
├── requirements.txt                # 依赖列表
├── .env                            # 环境变量（API Key，已 gitignore）
└── README.md
```

## 快速开始

### 环境依赖

- Python 3.10+
- Elasticsearch 8.x（需安装 `ik_max_word` 中文分词器）
- Redis（需支持向量检索）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/Stars0113/Agentic-RAG-Project.git
cd Agentic-RAG-Project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
# 在项目根目录创建 .env 文件，填入：
# DASHSCOPE_API_KEY=sk-your-api-key-here

# 4. 启动 ES 和 Redis 服务

# 5. 运行
python main.py
```

### 诊断检索质量

```bash
# 运行诊断脚本，检查各阶段召回和过滤情况
python test_retrieval.py
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 向量模型 | BAAI/bge-large-zh |
| 重排序模型 | BAAI/bge-reranker-large |
| 大模型 | 通义千问 qwen-turbo（DashScope API） |
| 搜索引擎 | Elasticsearch 8.x |
| 缓存 | Redis |
| 框架 | LangChain |
| 评估 | Ragas |

## 运行效果

三个测试查询的检索结果：

| 查询 | Top-1 召回结果 |
|------|---------------|
| 感冒了怎么办 | 感冒初期多喝温水、保证充足睡眠通常可以自愈。如果体温超过38.5度... |
| 高血压的饮食注意事项 | 高血压患者日常饮食应以低盐低脂为主，每日食盐摄入量不建议超过5克... |
| 糖尿病症状 | 糖尿病的典型症状是三多一少，即吃得多、饮得多、尿得多，但体重却减轻... |

## License

MIT License
