@echo off
chcp 65001 >nul
echo ==============================================
echo 正在启动企业级 Agentic RAG 医疗知识库...
echo ==============================================
echo 当前目录: %cd%
cd /d "E:\就业\ai应用开发agent开发\Enterprise_level_projects\chaifen_demo"
echo 切换到项目目录: %cd%
echo.

echo 检查 Python 是否可用...
python --version
if errorlevel 1 (
    echo ❌ Python 不可用，请检查环境变量
    pause
    exit /b 1
)

echo.
echo 启动 Streamlit 应用...
echo.
python -m streamlit run app/mian.py

echo.
echo 应用已退出
pause