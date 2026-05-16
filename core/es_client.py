# ES 客户端（连接、检索、插入）

# ES客户端（连接、插入、检索）
import time
from elasticsearch import Elasticsearch
from config.es_config import ES_HOST, ES_INDEX_NAME, ES_MAPPING, ES_TOP_N
from config.path_config import KB_FILE_PATH
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def load_knowledge_base(file_path):
    """使用 LangChain 加载并分块文档，替代 exec()"""
    # 根据文件类型选择加载器
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')

    # 加载文档
    docs = loader.load()

    # 自动分块（每块 500 字符，重叠 50 字符）
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 转换成原来需要的格式：[{"text": chunk}]
    return [{"text": chunk.page_content} for chunk in chunks]

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