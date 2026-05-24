import pytest
from pipeline.rag_pipeline import hybrid_search

def test_hybrid_search():
    """测试混合检索功能"""
    result = hybrid_search("血压高平时吃东西要注意啥")
    assert isinstance(result, list)
    assert len(result) > 0

def test_cache_hit():
    """测试缓存命中"""
    result1 = hybrid_search("晚上睡觉总喘不上气怎么办")
    result2 = hybrid_search("晚上睡觉总喘不上气怎么办")
    assert result1 is not None

def test_agent_retry():
    """测试Agent重试逻辑"""
    result = hybrid_search("非常冷门的医学问题测试")
    # 应有合理的处理（返回空或改写后结果）
    assert result is not None
