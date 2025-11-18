import json

# 读取 耳鼻喉科.json 文件
with open('耳鼻喉科.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理数据
new_data = []
for item in data:
    new_item = {
        "input": item["input_check"],
        "input_fy": "",
        "output": item["output_check"],
        "output_fy": ""
    }
    new_data.append(new_item)

# 将处理后的数据写入 耳鼻喉科1.json 文件
with open('耳鼻喉科1.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=2)

print("转换完成，已生成 耳鼻喉科1.json 文件。")