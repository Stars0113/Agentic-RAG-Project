# ES 客户端（连接、检索、插入）

# ES客户端（连接、插入、检索）
import time
from elasticsearch import Elasticsearch
from config.es_config import ES_HOST, ES_INDEX_NAME, ES_MAPPING, ES_TOP_N
from config.path_config import KB_FILE_PATH

def init_es_client():
    """初始化ES连接"""
    es = Elasticsearch(ES_HOST)
    print("等待 ES 连接...")
    for i in range(10):
        if es.ping():
            print("✅ ES 连接成功")
            break
        time.sleep(1)
    else:
        print("❌ ES 未启动")
        exit(1)
    # 重建索引
    if es.indices.exists(index=ES_INDEX_NAME):
        es.indices.delete(index=ES_INDEX_NAME)
    es.indices.create(index=ES_INDEX_NAME, body=ES_MAPPING)
    return es

def load_knowledge_base():
    """加载知识库文档"""
    with open(KB_FILE_PATH, 'r', encoding='utf-8') as f:
        file_content = f.read()
    exec_globals = {}
    exec(file_content, exec_globals)
    docs = exec_globals.get('docs',[])
    return docs

def insert_docs_to_es(es, docs, embedding_model):
    """将文档+向量插入ES"""
    texts = [doc["text"] for doc in docs]
    vectors = embedding_model.embed_documents(texts)
    for i in range(len(docs)):
        es.index(
            index=ES_INDEX_NAME,
            id=i,
            document={"content": docs[i]["text"], "vector": vectors[i]}
        )
    es.indices.refresh(index=ES_INDEX_NAME)
    print(f"✅ {len(docs)}条文档已插入ES")

def get_bm25_results(es, keyword, top_n=ES_TOP_N):
    """BM25检索"""
    res = es.search(index=ES_INDEX_NAME, query={"match": {"content": keyword}}, size=top_n)
    return [hit["_source"]["content"] for hit in res["hits"]["hits"][:top_n]]

def get_vector_results(es, query_vector, top_n=ES_TOP_N):
    """向量检索"""
    res = es.search(
        index=ES_INDEX_NAME,
        knn={"field": "vector", "query_vector": query_vector, "k": top_n, "num_candidates": top_n},
        size=top_n
    )
    return [hit["_source"]["content"] for hit in res["hits"]["hits"][:top_n]]