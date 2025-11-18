import pandas as pd
import re
import hashlib

class DataCleaner:
    def __init__(self):
        self.cleaned_data = []
    
    def remove_duplicates(self, data):
        """基于内容哈希去重"""
        seen_hashes = set()
        unique_data = []
        
        for item in data:
            # 创建内容哈希
            content = f"{item['question_zh']}{item['answer_zh']}"
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_data.append(item)
        
        return unique_data
    
    def clean_text(self, text):
        """清理文本"""
        if not text:
            return ""
        
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\w\s\.\,\?\!]', '', text)
        
        return text.strip()
    
    def filter_quality_data(self, data, min_question_length=5, min_answer_length=10):
        """过滤低质量数据"""
        filtered_data = []
        
        for item in data:
            question = item.get('question_zh', '')
            answer = item.get('answer_zh', '')
            
            # 检查长度要求
            if (len(question) >= min_question_length and 
                len(answer) >= min_answer_length and
                '广告' not in question and '广告' not in answer):
                
                # 清理文本
                item['question_zh'] = self.clean_text(question)
                item['answer_zh'] = self.clean_text(answer)
                
                filtered_data.append(item)
        
        return filtered_data
    
    def process_data(self, input_file, output_file):
        """处理数据"""
        # 读取数据
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file, encoding='utf-8-sig')
        else:  # json
            df = pd.read_json(input_file, encoding='utf-8')
        
        data = df.to_dict('records')
        
        # 数据清洗流程
        print(f"原始数据量: {len(data)}")
        
        # 1. 去重
        data = self.remove_duplicates(data)
        print(f"去重后数据量: {len(data)}")
        
        # 2. 质量过滤
        data = self.filter_quality_data(data)
        print(f"质量过滤后数据量: {len(data)}")
        
        # 保存清洗后的数据
        cleaned_df = pd.DataFrame(data)
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"清洗后的数据已保存到: {output_file}")
        
        return data

# 使用示例
if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.process_data('dxy_medical_qa.csv', 'dxy_medical_qa_cleaned.csv')