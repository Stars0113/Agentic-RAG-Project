# Ragas评估相关

# Ragas评估工具
from datasets import Dataset
from ragas.metrics import faithfulness, context_recall
from ragas import evaluate


def grade(score):
    if score >= 0.9:
        return "优秀", "答案内容完全来自检索文档，几乎无幻觉"
    elif score >= 0.8:
        return "良好", "答案基本忠于文档，偶有轻微发挥"
    elif score >= 0.7:
        return "合格", "答案大部分有依据，存在少量幻觉"
    else:
        return "不合格", "答案存在明显幻觉或严重遗漏，需要优化"


def generate_ragas_report(query, contexts, llm):
    if not contexts:
        print("⚠️ 无检索文档，跳过评估")
        return "未找到相关信息"

    # 生成答案
    context_text = "\n".join(contexts[:5])
    prompt = f"""你是一个专业的医疗健康助手。请根据以下参考文档回答用户问题。
如果文档中没有相关信息，请如实说"暂未找到相关信息"，不要瞎编。

参考文档：
{context_text}

用户问题：{query}

请给出专业、准确的回答："""
    answer = llm.invoke(prompt).strip()
    print("千问生成答案成功")

    # 构造ground_truth（可后续抽离到配置/数据文件）
    ground_truth = """晚上睡觉喘不上气的原因包括：
    1.睡姿不当，平躺压迫胸腔，建议侧卧；
    2.肥胖人群腹部脂肪压迫心肺；
    3.卧室空气浑浊缺氧，建议开窗通风；
    4.心肺功能偏弱，夜间回心血量增加加重负担；
    5.过敏性鼻炎导致鼻塞张口呼吸；
    6.胃食管反流刺激气道痉挛；
    7.睡眠呼吸暂停综合征；
    8.长期焦虑导致植物神经紊乱。建议保持侧卧、睡前不吃太饱、保持室内通风、适当减重，若频繁出现应就医检查心肺功能。"""

    # 构造数据集
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts[:5]],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)

    # 运行评估
    print("\n" + "=" * 70)
    print("🔥 开始 RAG 评估（Ragas）")
    print("=" * 70)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall],
        llm=llm
    )

    # 生成报告
    faith = result['faithfulness'][0]
    recall = result['context_recall'][0]
    f_level, f_desc = grade(faith)
    r_level, r_desc = grade(recall)
    avg = (faith + recall) / 2

    if avg >= 0.85:
        conclusion = "本次 Agentic RAG 系统检索结果质量优秀，过滤有效、内容准确、无幻觉，符合医疗健康场景要求。"
    elif avg >= 0.7:
        conclusion = "本次 Agentic RAG 系统检索结果质量合格，但仍有优化空间，建议调整 prompt 或检索策略。"
    else:
        conclusion = "本次 Agentic RAG 系统检索结果质量不合格，存在明显幻觉或信息遗漏，需要重点排查检索质量和 prompt 设置。"

    report = f"""
    ==================== RAG 检索效果评估报告 ====================

    评估模型：Ragas（v0.1.10）
    评估指标：Faithfulness（忠实度）、Context Recall（上下文召回率）
    用户问题：{query}

    1. 忠实度（Faithfulness）：{faith:.2f}  [{f_level}]
       - 说明：{f_desc}

    2. 上下文召回率（Context Recall）：{recall:.2f}  [{r_level}]
       - 说明：{r_desc}

    综合结论：
    {conclusion}
    """
    return report