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
        prompt = f"""请判断以下内容是否能回答用户的问题。

用户问题：{query}

候选内容：{doc}

请严格按以下JSON格式输出，不要输出任何其他内容：
{{"relevant": true或false, "confidence": 0.0到1.0之间的数字, "reason": "一句话说明原因"}}"""
        res = llm.invoke(prompt)
        #  robust JSON解析：从返回文本中提取JSON
        try:
            import re
            match = re.search(r'\{[^{}]*"relevant"[^{}]*\}', res, re.DOTALL)
            if match:
                j = json.loads(match.group())
            else:
                j = json.loads(res)
            if j.get("relevant") and j.get("confidence", 0) >= L2_CONFIDENCE_THRESHOLD:
                valid.append(doc)
        except Exception as e:
            # 解析失败，打印日志便于调试，默认保留（保守策略）
            print(f"  ⚠️ L2解析失败，保留文档: {str(e)}")
            valid.append(doc)
    print(f"✅ L2评估完成：合格{len(valid)}条 / 共{len(docs)}条")
    return valid