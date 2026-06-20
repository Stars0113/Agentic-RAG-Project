# 检索质量诊断脚本 - 在本地运行
# 用法：cd 到 chaifen_demo 目录，然后 python test_retrieval.py

import sys
sys.path.insert(0, ".")

from config.model_config import EMBEDDING_MODEL_NAME
from langchain_huggingface import HuggingFaceEmbeddings
from core.es_client import init_es_client, load_knowledge_base, insert_docs_to_es
from core.retriever import hybrid_retrieve
from utils.filter_utils import l0_filter, l1_emb_filter, l2_nli_eval
from core.agent import agent_judge
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 初始化
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"local_files_only": True})
es = init_es_client()

# 检查ES里有没有数据
from config.es_config import ES_INDEX_NAME as INDEX_NAME
count = es.count(index=INDEX_NAME)
print(f"ES中文档数量: {count['count']}")
if count['count'] == 0:
    print("⚠️ ES为空，先加载知识库...")
    from config.path_config import KB_FILE_PATH
    docs = load_knowledge_base(KB_FILE_PATH)
    insert_docs_to_es(es, docs, embedding)
    print(f"已插入 {len(docs)} 条文档")

# 测试查询
test_queries = [
    "感冒了怎么办",
    "高血压的饮食注意事项",
    "糖尿病症状",
]

llm = None  # 延迟初始化，避免没配API Key时报错
from core.llm_client import init_llm

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"查询: {query}")
    print(f"{'='*50}")

    # 1. 混合检索
    bm25_list, vec_list, rrf_result, final_result = hybrid_retrieve(query, es, embedding, top_n=20)

    print(f"\n[检索阶段]")
    print(f"  BM25召回: {len(bm25_list)} 条")
    print(f"  向量召回: {len(vec_list)} 条")
    print(f"  RRF融合后: {len(rrf_result)} 条")
    print(f"  重排序后: {len(final_result)} 条")

    # 显示前3条结果预览
    print(f"\n  重排序前3条预览:")
    for i, doc in enumerate(final_result[:3]):
        print(f"    [{i+1}] {doc[:80]}...")

    # 2. 过滤
    after_l0 = l0_filter(final_result)
    strong_docs, edge_docs = l1_emb_filter(query, after_l0, embedding)

    print(f"\n[过滤阶段]")
    print(f"  L0过滤后: {len(after_l0)} 条")
    print(f"  L1强相关: {len(strong_docs)} 条")
    print(f"  L1边缘相关: {len(edge_docs)} 条")

    if len(edge_docs) > 0:
        print(f"\n  边缘文档预览（前2条）:")
        for i, doc in enumerate(edge_docs[:2]):
            print(f"    [{i+1}] {doc[:80]}...")

    # 3. L2评估（需要LLM）
    if llm is None:
        try:
            llm = init_llm()
            print("\n✅ LLM初始化成功")
        except Exception as e:
            print(f"\n⚠️ LLM初始化失败（跳过L2）: {e}")
            llm = "FAILED"

    if llm != "FAILED":
        edge_valid = l2_nli_eval(query, edge_docs, llm)
        print(f"\n[L2评估阶段]")
        print(f"  边缘文档经L2评估后合格: {len(edge_valid)} 条")

        # Agent决策
        status, final_docs = agent_judge(strong_docs, edge_valid)
        print(f"\n[Agent决策]")
        print(f"  状态: {status}")
        print(f"  最终文档数: {len(final_docs) if final_docs else 0}")

        if final_docs:
            print(f"\n  最终文档预览:")
            for i, doc in enumerate(final_docs[:3]):
                print(f"    [{i+1}] {doc[:80]}...")
    else:
        print("\n⚠️ 跳过L2评估和Agent决策（LLM未初始化）")

print("\n\n✅ 诊断完成！")
print("如果某一步结果异常（比如L0后数量为0，或L2后数量为0），")
print("请检查对应的阈值配置（config/model_config.py）")
