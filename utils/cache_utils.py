# Redis 语义缓存相关

# Redis缓存工具
import time
import numpy as np
from redis import Redis
from config.redis_config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_CACHE_INDEX, REDIS_CACHE_PREFIX, REDIS_CACHE_EXPIRE, REDIS_CACHE_THRESHOLD
from utils.vector_utils import vec_to_bytes

# 初始化Redis客户端（单例）
_redis_client = None
def init_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
        # 创建索引（忽略已存在）
        try:
            _redis_client.execute_command(f"""
                FT.CREATE {REDIS_CACHE_INDEX} ON HASH PREFIX 1 {REDIS_CACHE_PREFIX}
                SCHEMA question TEXT answer TEXT vector VECTOR FLAT 6 TYPE FLOAT32 DIM 1024 DISTANCE_METRIC COSINE
            """)
            print("✅ Redis 缓存索引创建成功")
        except Exception as e:
            pass
    return _redis_client

def get_redis_cache(question, embedding_model, threshold=REDIS_CACHE_THRESHOLD):
    redis_client = init_redis_client()
    start_time = time.time()
    vec = embedding_model.embed_query(question)
    vec_bytes = vec_to_bytes(vec)
    results = redis_client.ft(REDIS_CACHE_INDEX).search(
        "*=>[KNN 1 @vector $vec AS score]", {"vec": vec_bytes}
    )
    if results.docs:
        score = 1 - float(results.docs[0].score)
        if score >= threshold:
            print(f"  💰 [缓存命中] 耗时: {(time.time()-start_time)*1000:.1f}ms, 相似度: {score:.2f}")
            return True, results.docs[0].answer
    return False, None

def set_redis_cache(question, answer, embedding_model):
    redis_client = init_redis_client()
    vec = embedding_model.embed_query(question)
    vec_bytes = vec_to_bytes(vec)
    key = f"{REDIS_CACHE_PREFIX}{question}"
    redis_client.hset(key, mapping={"question": question, "answer": answer, "vector": vec_bytes})
    redis_client.expire(key, REDIS_CACHE_EXPIRE)