# 示例代码：外部脚本（如main.py）
# -*- coding: utf-8 -*-
import json
from translation import Translation

def process_file(filename):
    """处理单个JSON文件"""
    # 初始化翻译类
    translator = Translation()

    # 读取文件
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件 {filename} 失败: {e}")
        return

    # 遍历每个条目，翻译并填充字段
    for index, item in enumerate(data):
        try:

            # 翻译input字段（汉语→藏语）
##             input_zh = item.get("input", "").strip()
##             if input_zh:
##                 print(f"[{filename}] 正在翻译第 {index+1} 条 input...")
##                 input_tb = translator.translate(input_zh, src_lan="ZH", tgt_lan="TB", append_tag=True)
##                 item["input_fy"] = input_tb
            
            # 翻译output字段（汉语→藏语）
##             output_zh = item.get("output", "").strip()
##             if output_zh:
##                 print(f"[{filename}] 正在翻译第 {index+1} 条 output...")
##                 output_tb = translator.translate(output_zh, src_lan="ZH", tgt_lan="TB", append_tag=True)
##                 item["output_fy"] = output_tb

            # 翻译department字段（汉语→藏语）
            department_zh = item.get("department", "").strip()
            if department_zh:
                print(f"[{filename}] 正在翻译第 {index+1} 条 department...")
                department_tb = translator.translate(department_zh, src_lan="ZH", tgt_lan="TB", append_tag=True)
                item["department_fy"] = department_tb

        except Exception as e:
            print(f"[{filename}] 翻译第 {index+1} 条时发生错误: {e}")
            continue

    # 写回JSON文件
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[{filename}] 翻译完成，结果已保存")
    except Exception as e:
        print(f"[{filename}] 保存文件失败: {e}")

def main():
    datasets = ["text.json"]
    for filename in datasets:
        print(f"\n====== 开始处理 {filename} ======")
        process_file(filename)

if __name__ == "__main__":
    main()