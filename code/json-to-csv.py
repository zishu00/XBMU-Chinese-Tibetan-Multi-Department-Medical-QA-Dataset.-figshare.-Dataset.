# json_to_csv_utf8sig.py
import json
import csv
from pathlib import Path

json_path = Path(r"E:\Xbmu\AC\实验\数据集\儿科\05翻译\全部1\神经内科.json")   # 若路径不同，改这里或使用绝对路径
csv_path = Path(r"E:\Xbmu\AC\实验\数据集\儿科\05翻译\全部1\神经内科_answer_bo.csv")     # 输出文件

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

#headers = ["id", "question_zh", "question_bo", "answer_zh", "answer_bo", "category"]
headers = ["id", "answer_bo",  "category"]

# 使用 utf-8-sig 会在文件开头写入 BOM，方便 Excel 正确识别 UTF-8
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i, item in enumerate(data, start=1):
        writer.writerow([
            i,
            #item.get("input", ""),
            #item.get("input_fy", ""),
            #item.get("output", ""),
            item.get("output_fy", ""),
            "神经内科"
        ])

print(f"已生成：{csv_path}（编码：utf-8-sig）")
