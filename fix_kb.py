# 修复知识库格式脚本
# 用法：cd 到 chaifen_demo 目录，然后 python fix_kb.py

import re
import os

kb_path = "docs/medical_kb.txt"
output_path = "docs/medical_kb_clean.txt"

with open(kb_path, "r", encoding="utf-8") as f:
    content = f.read()

# 提取所有 "text": "..." 的内容
pattern = r'"text":\s*"([^"]+)"'
matches = re.findall(pattern, content)

print(f"找到 {len(matches)} 条记录")

# 写入纯文本格式（每行一条）
with open(output_path, "w", encoding="utf-8") as f:
    for text in matches:
        f.write(text + "\n")

print(f"已生成 {output_path}")
print(f"共 {len(matches)} 条，每行一条")
print("\n前3条预览：")
for i, text in enumerate(matches[:3]):
    print(f"  [{i+1}] {text}")
