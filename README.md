# 🏥 企业级 Agentic RAG 医疗知识库问答系统

基于 2026 主流 RAG 架构设计，实现**混合检索 + 三级渐进式过滤 + Agentic 自主决策重试 + 语义缓存**，彻底解决传统 RAG 系统“检索噪音大、无结果硬答、重复计算成本高”的痛点。

## 🌟 核心技术亮点（面试必问点）

1. **🔀 混合检索 + RRF 融合 + Cross-Encoder 重排序**
   - 并行执行 BM25（关键词精准匹配）与 Vector（语义泛化匹配），通过 RRF 算法融合双路结果，再接入 `bge-reranker-large` 交叉编码器精排，确保 Top5 结果极高相关性。
2. **🛡️ 企业级 L0-L2 三级渐进式过滤（核心差异化）**
   - **L0 规则过滤**：剔除黑名单词（广告/乱码）及超长超短无效文本。
   - **L1 向量过滤**：基于余弦相似度划分“强相关”与“边缘相关”，剔除语义偏差文档。
   - **L2 大模型 NLI 评估**：利用 LLM 判断文档是否真正具备回答问题的能力（置信度>0.7），彻底解决“语义相似但答非所问”的幻觉问题。
3. **🤖 Agentic 自主决策与查询改写**
   - 检索结果不足时，Agent 拒绝直接回答，自主判断 `LACK` 或 `SERIOUS_LACK`。
   - 自动调用 LLM 改写 Query 或补充领域关键词（如“功效 禁忌”），最多重试 3 次直至信息充足（`ENOUGH`）。
4. **⚡ Redis 语义缓存**
   - 基于 Vector 相似度匹配历史问答，相似度阈值>0.75 直接命中缓存，毫秒级响应，大幅降低 LLM 和检索引擎的并发压力。
5. **📊 Ragas 自动化评估体系**
   - 内置 `Faithfulness`（忠实度）与 `Context Recall`（上下文召回率）量化评估，生成带等级（优秀/良好/合格/不合格）的诊断报告，让 RAG 效果优化“有据可依”。

## 🏗️ 系统架构流程图



## 📁 项目工程化结构

项目采用严格的核心逻辑、配置、工具集解耦设计，具备高可维护性与可扩展性：
Agentic-RAG-Project/
├── core/ # 核心业务逻辑
│ ├── es_client.py # ES 连接与文档向量化入库
│ ├── llm_client.py # 千问大模型客户端
│ ├── retriever.py # 混合检索器编排
│ └── agent.py # Agent 决策与查询改写
├── config/ # 集中化配置管理
│ ├── es_config.py # ElasticSearch 配置
│ ├── model_config.py # 向量/重排/LLM 模型及阈值配置
│ ├── path_config.py # 文件路径配置
│ └── redis_config.py # Redis 缓存配置
├── utils/ # 通用工具集
│ ├── rank_utils.py # RRF 融合与重排序算法
│ ├── filter_utils.py # L0/L1/L2 三级过滤核心实现
│ ├── eval_utils.py # Ragas 评估与报告生成
│ └── cache_utils.py # Redis 语义缓存读写
├── rag_pipeline.py # 🌟 完整 RAG 流程统一编排入口
├── docs/ # 知识库原始文档
├── .env # 环境变量（含 API Key，已忽略上传）
└── README.md


## 🚀 快速开始

### 1. 环境依赖

- Python 3.10+
- Elasticsearch 8.x (需安装 `ik_max_word` 分词器)
- Redis Stack (需支持 RediSearch 向量检索)

### 2. 安装依赖包

pip install -r requirements.txt


### 3. 配置环境变量

在项目根目录创建 `.env` 文件，填入你的大模型 API Key：
DASHSCOPE_API_KEY=sk-your-api-key-here

### 4. 初始化知识库并启动

确保本地 ES 和 Redis 服务已启动
python rag_pipeline.py


## 🧪 自动化测试与评估

本项目深度融合自动化测试思维，提供标准化验证手段：

1. **Ragas 质量评估**：每次检索自动生成诊断报告，量化评估检索准确率与回答忠实度，确保检索命中率 $\ge$ 80%。
2. **单元测试覆盖**：对检索器、过滤器、缓存等核心模块提供基础测试用例，保障系统迭代稳定性，测试通过率 100%。

## 💡 未来优化方向 (V2.0)

- [ ] 接入 Streamlit 实现可视化交互界面，支持“上传-查询-回答”闭环
- [ ] 支持用户前端动态上传 PDF/TXT 文件并实时向量化
- [ ] 替换 ES 为轻量级 Qdrant，降低部署门槛
- [ ] 引入多智能体协作架构，进一步拆分评估与生成职责

## 📄 License

MIT License









