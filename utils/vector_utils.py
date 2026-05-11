# 向量计算（余弦相似度等）

# 向量相关工具
import numpy as np

def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2 + 1e-8)

def vec_to_bytes(vec):
    """将向量转为Redis存储的bytes格式"""
    return np.array(vec, dtype=np.float32).tobytes()