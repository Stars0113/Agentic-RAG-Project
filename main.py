# 入口文件（测试/运行）

# 项目入口
from pipeline.rag_pipeline import hybrid_search

# 测试集
test_queries = [
    "血压高平时吃东西要注意啥",
    "晚上睡觉总喘不上气怎么办",
    "腿脚发麻使不上劲是咋回事"
]

if __name__ == "__main__":
    # 第一次检索（无缓存）
    first_results = hybrid_search(test_queries[1])
    # 测试缓存命中
    first_results1 = hybrid_search("晚上睡觉总喘不上气怎么办")
    second_results1 = hybrid_search("睡觉的时候觉得喘不过气来咋整")