import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pipeline.rag_pipeline import hybrid_search, init_global_components
from config.path_config import KB_FILE_PATH

# 初始化全局组件（首次运行时）
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def init_components():
    if not st.session_state.initialized:
        init_global_components(KB_FILE_PATH)
        st.session_state.initialized = True


st.title("企业级 Agentic RAG 医疗知识库")
st.write("上传 PDF/TXT 文件，自动分块向量化，支持智能问答")

# 文件上传组件
uploaded_file = st.file_uploader("选择知识库文件（PDF/TXT）", type=["pdf", "txt"])

if uploaded_file is not None:
    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)

    # 保存上传的文件
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 重新初始化组件（加载新文件）
    init_global_components(file_path)
    st.success(f"已加载文件：{uploaded_file.name}")

# 查询输入
query = st.text_input("请输入您的医疗问题：")

if query:
    # 确保组件已初始化
    init_components()

    with st.spinner("Agent 正在思考、检索与评估..."):
        # 调用 RAG 流程
        result = hybrid_search(query)

    if result:
        st.markdown("### 🤖 回答依据")
        for doc in result:
            st.info(doc)

        st.markdown("### 📊 评估报告")
        report_path = os.path.join("data", "report.txt")  # ← 改这一行
        if os.path.exists(report_path):  # ← 加这两行
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.text(report_content)
        else:
            st.warning("评估报告尚未生成")