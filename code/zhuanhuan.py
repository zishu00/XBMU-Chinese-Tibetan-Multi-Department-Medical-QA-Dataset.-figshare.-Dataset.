import json

# 读取JSON文件
with open('text-TC.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理每个条目
for item in data:
    # 交换input和output的内容
    input_content = item["input"]
    output_content = item["output"]
    
    item["input"] = output_content
    item["output"] = input_content
    
    # 修改instruction字段
    item["instruction"] = "现在你是一个语言专家，请根据汉语内容翻译成藏语："

# 保存修改后的数据
with open('text-CT.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("处理完成！已保存为新的JSON文件。")