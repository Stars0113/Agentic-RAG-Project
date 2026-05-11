# Agent决策/查询改写

# Agent决策逻辑
def agent_judge(strong, edge_valid):
    """Agent决策：判断文档是否充足"""
    total = len(strong) + len(edge_valid)
    if total >= 5:
        return "ENOUGH", strong + edge_valid
    elif 2 <= total <= 4:
        return "LACK", None
    else:
        return "SERIOUS_LACK", None

def rewrite_query(query, llm):
    """Agent自动改写查询"""
    prompt = f"""
    原问题：{query}
    检索结果不足，请生成更精准的查询。
    输出1个查询即可。
    """
    return llm.invoke(prompt).strip()