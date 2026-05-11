# L0/L1/L2过滤相关

# L0/L1/L2 过滤逻辑
import numpy as np
import json
from config.model_config import L0_MIN_LEN, L0_MAX_LEN, L0_BLACK_LIST, L1_STRONG_THRESHOLD, L1_EDGE_THRESHOLD, L2_CONFIDENCE_THRESHOLD
from utils.vector_utils import cosine_sim

def l0_filter(docs):
    filtered = []
    for doc in docs:
        if len(doc) < L0_MIN_LEN or len(doc) > L0_MAX_LEN:
            continue
        if any(key in doc for key in L0_BLACK_LIST):
            continue
        filtered.append(doc)
    print(f"✅ L0过滤完成：原{len(docs)}条 → 剩余{len(filtered)}条")
    return filtered

def l1_emb_filter(query, docs, embedding_model):
    q_vec = embedding_model.embed_query(query)
    strong = []
    edge = []
    for doc in docs:
        doc_vec = embedding_model.embed_documents([doc])[0]
        sim = cosine_sim(q_vec, doc_vec)
        if sim > L1_STRONG_THRESHOLD:
            strong.append(doc)
        elif sim >= L1_EDGE_THRESHOLD:
            edge.append(doc)
    print(f"✅ L1过滤完成：强相关{len(strong)}条，边缘{len(edge)}条")
    return strong, edge

def l2_nli_eval(query, docs, llm):
    valid = []
    for doc in docs:
        prompt = f"""
        问题：{query}
        内容：{doc}
        请判断内容是否能回答问题，仅输出JSON：
        {{"relevant":true/false,"confidence":0.8,"reason":"..."}}
        """
        res = llm.invoke(prompt)
        try:
            j = json.loads(res)
            if j["relevant"] and j["confidence"] >= L2_CONFIDENCE_THRESHOLD:
                valid.append(doc)
        except:
            continue
    print(f"✅ L2评估完成：合格{len(valid)}条")
    return valid