#调取api对数据进行清洗
import json
import os
import time
from typing import List, Dict
from zhipuai import ZhipuAI

class MedicalDataChecker:
    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)
        self.file_list = ["text1.json"]  # 预留后续文件位置
        os.makedirs("check_results", exist_ok=True)

    def load_json_data(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"加载文件失败：{e}")
            return []

    def generate_prompt(self, content: str, field_type: str) -> str:
        if field_type == "input":
            return f"""请保留以下患者提问原始语义，仅输出修改结果：
内容：{content}
修改要点：修改结果保留原始语义、语句通顺无病句、存在逻辑性错误。另外，如果内容不是涉及儿科医学问题的提问，请直接输出“无”。
请直接给出具体的修改结果（如果需要），无需额外判断语句。"""
        else:
            return f"""请保留以下医生提问原始语义，仅输出修改结果：
内容：{content}
修改要点：修改结果保留原始语义、语句通顺无病句、存在逻辑性错误、医学知识准确可靠（从儿科专业角度）。另外，如果内容不是涉及儿科医学问题的回答，请直接输出“无”。
请直接给出具体的修改结果（如果需要），无需额外判断语句。"""

    def check_stream(self, content: str, field_type: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="glm-4.5-air",
                messages=[{"role": "user", "content": self.generate_prompt(content, field_type)}],
                stream=True
            )
            full_res = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_res.append(chunk.choices[0].delta.content)
            return ''.join(full_res)
        except Exception as e:
            return f"流式检测出错：{e}"

    def check_normal(self, content: str, field_type: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="glm-4.5-air",
                messages=[{"role": "user", "content": self.generate_prompt(content, field_type)}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"非流式检测出错：{e}"

    def process_file(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"文件不存在：{file_path}")
            return
        data = self.load_json_data(file_path)
        if not data:
            return
        results = []
        for i, item in enumerate(data):
            print(f"处理第{i+1}条")
            results.append({
                "input": item.get("input", ""),
                "input_check": self.check_normal(item.get("input", ""), "input"),
                "output": item.get("output", ""),
                "output_check": self.check_stream(item.get("output", ""), "output")
            })
            time.sleep(1)
        output_file = f"check_results/{os.path.splitext(os.path.basename(file_path))[0]}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"{file}结果已保存至：{output_file}")

if __name__ == "__main__":
    # 替换为您的API密钥
    API_KEY = "KEY"
    checker = MedicalDataChecker(API_KEY)
    # 处理文件
    checker.process_file("text1.json")
    # 如需添加后续文件，可使用：
    #checker.file_list.extend(["神经内科_10000.json", "外科_6350.json"])
    #for file in checker.file_list:
    #    checker.process_file(file)