# Redis 缓存配置

# Redis 缓存配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_CACHE_INDEX = "chaifen_demo_idx_rag_cache"
REDIS_CACHE_PREFIX = "chaifen_demo:rag:cache:"
REDIS_CACHE_EXPIRE = 86400  # 缓存过期时间（秒）
REDIS_CACHE_THRESHOLD = 0.75  # 缓存命中相似度阈值