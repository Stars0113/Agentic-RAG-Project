# 混合检索（BM25+向量+RRF）

# 混合检索器
from utils.rank_utils import rrf_fusion, rerank
from core.es_client import get_bm25_results, get_vector_results

def hybrid_retrieve(query, es, embedding_model, top_n=50):
    """混合检索：BM25+向量+RRF+重排序"""
    query_vector = embedding_model.embed_query(query)
    # 1. 基础检索
    bm25_list = get_bm25_results(es, query)
    vec_list = get_vector_results(es, query_vector)
    # 2. RRF融合
    rrf_result = rrf_fusion(bm25_list, vec_list)
    # 3. 重排序
    final_result = rerank(query, rrf_result, top_n=top_n)
    return bm25_list, vec_list, rrf_result, final_result