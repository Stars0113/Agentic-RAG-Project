# 大模型客户端（千问调用）

# 大模型客户端
import os
from langchain_community.llms import Tongyi
from config.model_config import DASHSCOPE_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE

def init_llm():
    """初始化千问大模型"""
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    llm = Tongyi(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE
    )
    print("✅ 千问大模型加载完成")
    return llm