import requests
import json
import time

base_url = "http://127.0.0.1:6006/"  # ⚠️替换为你自己的地址
model_name = "chatglm3-6b"

headers = {"Content-Type": "application/json"}

def call_api(messages):
    data = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 512
    }
    response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("API错误:", response.status_code, response.text)
        return None

def augment_sample(input_text, num=20):
    prompt = f"请你根据以下内容，扩展生成{num}条内容相似、主题仍与“内科”相关的中文句子：\n{input_text}"
    messages = [{"role": "user", "content": prompt}]
    response = call_api(messages)
    return response

def translate_to_english(chinese_sentence):
    prompt = f"请你将以下汉语语句翻译为英语:\n{chinese_sentence}"
    messages = [{"role": "user", "content": prompt}]
    response = call_api(messages)
    return response

# 加载原始数据（你可替换为 text.py 中导入）
with open("内科_f.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

augmented_data = []

for item in raw_data:
    base_input = item["input"]
    print(f"正在扩展：{base_input}")
    expansion_result = augment_sample(base_input, num=20)

    if not expansion_result:
        continue

    # 拆分返回的中文句子
    candidates = [line.strip("1234567890.、-：: ") for line in expansion_result.strip().split("\n") if line.strip()]
    
    for zh in candidates:
        en = translate_to_english(zh)
        if not en:
            continue
        augmented_data.append({
            "instruction": item["instruction"],
            "input": zh,
            "output": en
        })
        print("生成：", zh, "→", en)
        time.sleep(1)

# 保存增强结果
with open("内科_z.json", "w", encoding="utf-8") as f:
    json.dump(augmented_data, f, ensure_ascii=False, indent=2)

print(f"完成，总共生成 {len(augmented_data)} 条增强数据。")
