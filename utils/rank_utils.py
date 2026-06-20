# RRF融合/重排序相关

# RRF融合 + 重排序
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config.model_config import RERANK_MODEL_PATH, RRF_K

def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (rank + k)

def rrf_fusion(bm25_list: list, vector_list: list, k: int = RRF_K) -> list:
    score_dict = {}
    for idx, doc in enumerate(bm25_list):
        rank = idx + 1
        score_dict[doc] = score_dict.get(doc, 0.0) + rrf_score(rank, k)
    for idx, doc in enumerate(vector_list):
        rank = idx + 1
        score_dict[doc] = score_dict.get(doc, 0.0) + rrf_score(rank, k)
    sorted_result = sorted(score_dict.items(), key=lambda x: -x[1])
    return [item[0] for item in sorted_result[:100]]

# 初始化重排序模型（单例）
_tokenizer = None
_rerank_model = None
def init_rerank_model():
    global _tokenizer, _rerank_model
    if _tokenizer is None or _rerank_model is None:
        print("加载重排序模型 bge-reranker-large...")
        _tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH, local_files_only=True)
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_PATH, local_files_only=True)
        _rerank_model.eval()
    return _tokenizer, _rerank_model

def rerank(query: str, docs: list, top_n: int = 50) -> list:
    tokenizer, rerank_model = init_rerank_model()
    scores = []
    for doc in docs:
        inputs = tokenizer(query, doc, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = rerank_model(** inputs)
        score = outputs.logits.item()
        scores.append((doc, score))
    scores_sorted = sorted(scores, key=lambda x: -x[1])
    return [item[0] for item in scores_sorted[:top_n]]