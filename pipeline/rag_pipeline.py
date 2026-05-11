# 完整RAG流程（缓存→检索→过滤→决策→评估）

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings
from config.model_config import EMBEDDING_MODEL_NAME
from config.path_config import OUTPUT_DIR
from core.es_client import init_es_client, load_knowledge_base, insert_docs_to_es
from core.llm_client import init_llm
from core.retriever import hybrid_retrieve
from core.agent import agent_judge, rewrite_query
from utils.cache_utils import get_redis_cache, set_redis_cache
from utils.filter_utils import l0_filter, l1_emb_filter, l2_nli_eval
from utils.eval_utils import generate_ragas_report

# 初始化全局组件（仅初始化一次）
_es = None
_embedding = None
_llm = None


def init_global_components():
    global _es, _embedding, _llm
    # 向量模型
    _embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    # ES
    _es = init_es_client()
    docs = load_knowledge_base()
    insert_docs_to_es(_es, docs, _embedding)
    # 大模型
    _llm = init_llm()


def hybrid_search(query: str, top_n: int = 50) -> list:
    """完整RAG流程入口"""
    # 初始化组件（首次调用时）
    if _es is None or _embedding is None or _llm is None:
        init_global_components()

    # 1. 检查缓存
    is_hit, cached_answer = get_redis_cache(query, _embedding)
    if is_hit:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "report.txt"), 'w', encoding='utf-8') as out:
            out.write(cached_answer)
        print("🚀 跳过全部检索与评估流程，直接返回！\n")
        return ["[来自缓存]"]

    # 2. Agent重试逻辑（最多3次）
    current_query = query
    for attempt in range(3):
        # 混合检索
        bm25_list, vec_list, rrf_result, final_result = hybrid_retrieve(current_query, _es, _embedding, top_n)
        # 过滤
        after_l0 = l0_filter(final_result)
        strong_docs, edge_docs = l1_emb_filter(current_query, after_l0, _embedding)
        edge_valid = l2_nli_eval(current_query, edge_docs, _llm)
        # Agent决策
        status, final_docs = agent_judge(strong_docs, edge_valid)
        # 生成评估报告
        report = generate_ragas_report(current_query, final_docs, _llm)

        if status == "ENOUGH":
            print("\n✅ Agent 判定：内容充足！")
            # 保存结果
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            results = {
                "bm25_results.txt": bm25_list,
                "vec_results.txt": vec_list,
                "rrf_results.txt": rrf_result,
                "final_results.txt": final_result,
                "after_l0.txt": after_l0,
                "strong_docs.txt": strong_docs,
                "edge_docs.txt": edge_docs,
                "final_docs.txt": final_docs,
                "report.txt": report,
            }
            for filename, data in results.items():
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w', encoding='utf-8') as out:
                    if filename == "report.txt":
                        out.write(data)
                    else:
                        out.write("\n".join(data))
            print(f"✅ {len(results)}个结果文件已保存")
            # 存入缓存
            set_redis_cache(query, report, _embedding)
            print("💾 已将本次结果存入 Redis 缓存")
            return final_docs[:5]

        elif status == "LACK":
            print("\n⚠️ 内容不足，Agent 自动改写查询...")
            current_query = rewrite_query(current_query, _llm)

        else:
            print("\n❌ 严重不足，扩展关键词重搜...")
            current_query = current_query + " 功效 作用 营养价值 禁忌"

    # 重试失败兜底
    print("\n❌ Agent 多次重试失败，未找到足够文档。")
    return []