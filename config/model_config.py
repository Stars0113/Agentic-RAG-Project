# 向量/重排序/大模型配置

# 模型相关配置
import os
from dotenv import load_dotenv

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 向量模型
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh"
# 重排序模型
RERANK_MODEL_PATH = "BAAI/bge-reranker-large"
# 大模型（千问）
LLM_MODEL_NAME = "qwen-turbo"
LLM_TEMPERATURE = 0.1
# RRF参数
RRF_K = 100
# 过滤参数
L0_MIN_LEN = 20
L0_MAX_LEN = 10000
L0_BLACK_LIST = ["广告", "营销", "直播间", "乱码", "跳转"]
L1_STRONG_THRESHOLD = 0.8
L1_EDGE_THRESHOLD = 0.5
L2_CONFIDENCE_THRESHOLD = 0.7