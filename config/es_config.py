# ElasticSearch 连接配置
ES_HOST = "http://localhost:9200"
ES_INDEX_NAME = "chaifen_demo"
ES_MAPPING = {
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
            "vector": {"type": "dense_vector", "dims": 1024}
        }
    }
}
ES_TOP_N = 100  # 检索默认返回条数