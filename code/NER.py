import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import re
from collections import defaultdict

class NERSensitiveInfoDetector:
    """使用命名实体识别检测和替换敏感信息"""
    
    def __init__(self, model_name='bert-base-chinese'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 加载NER模型（这里使用通用的中文BERT模型，实际应用中可以使用医学领域微调的模型）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            
            # 创建NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple"
            )
            
        except Exception as e:
            self.logger.error(f"加载NER模型失败: {e}")
            self.ner_pipeline = None
        
        # 实体类型映射和替换策略
        self.entity_replacement_strategy = {
            'PER': '某某',      # 人名 -> 某某
            'LOC': '某地',      # 地点 -> 某地
            'ORG': '某机构',    # 组织 -> 某机构
            'GPE': '某地',      # 地理政治实体 -> 某地
            'FAC': '某场所',    # 设施 -> 某场所
            'DATE': '某时间',   # 日期 -> 某时间
        }
        
        # 对于中文NER，可能需要调整标签映射
        self.label_mapping = {
            'B-PER': 'PER', 'I-PER': 'PER',
            'B-LOC': 'LOC', 'I-LOC': 'LOC', 
            'B-ORG': 'ORG', 'I-ORG': 'ORG',
            'B-GPE': 'GPE', 'I-GPE': 'GPE',
            'B-FAC': 'FAC', 'I-FAC': 'FAC',
            'B-DATE': 'DATE', 'I-DATE': 'DATE',
        }
        
        # 补充正则模式（用于NER可能漏检的情况）
        self.backup_patterns = {
            'PHONE': r'1[3-9]\d{9}',
            'ID_CARD': r'[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]',
            'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'BANK_CARD': r'\d{16,19}'
        }
    
    def detect_entities(self, text: str) -> List[Dict]:
        """使用NER模型检测实体"""
        if not self.ner_pipeline or not text.strip():
            return []
        
        try:
            entities = self.ner_pipeline(text)
            return entities
        except Exception as e:
            self.logger.error(f"NER检测失败: {e}")
            return []
    
    def backup_regex_detection(self, text: str) -> List[Dict]:
        """使用正则表达式进行补充检测"""
        entities = []
        
        for entity_type, pattern in self.backup_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'entity_group': entity_type,
                    'score': 0.95,  # 正则匹配给予高置信度
                    'word': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    def merge_entities(self, ner_entities: List[Dict], regex_entities: List[Dict]) -> List[Dict]:
        """合并NER和正则表达式检测结果，去除重叠"""
        all_entities = ner_entities + regex_entities
        
        if not all_entities:
            return []
        
        # 按起始位置排序
        all_entities.sort(key=lambda x: x['start'])
        
        # 去除重叠实体（保留置信度高的）
        merged_entities = []
        current_entity = all_entities[0]
        
        for entity in all_entities[1:]:
            if entity['start'] < current_entity['end']:
                # 有重叠，保留置信度高的
                if entity['score'] > current_entity['score']:
                    current_entity = entity
            else:
                merged_entities.append(current_entity)
                current_entity = entity
        
        merged_entities.append(current_entity)
        
        return merged_entities
    
    def replace_entities(self, text: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
        """替换检测到的实体"""
        if not entities:
            return text, []
        
        # 按起始位置逆序排序，以便从后往前替换
        entities.sort(key=lambda x: x['start'], reverse=True)
        
        replaced_text = text
        replacement_log = []
        
        for entity in entities:
            entity_type = entity['entity_group']
            original_word = entity['word']
            start, end = entity['start'], entity['end']
            
            # 获取替换策略
            replacement = self.entity_replacement_strategy.get(entity_type, '某')
            
            # 执行替换
            replaced_text = replaced_text[:start] + replacement + replaced_text[end:]
            
            # 记录替换信息
            replacement_log.append({
                'entity_type': entity_type,
                'original': original_word,
                'replacement': replacement,
                'position': (start, end),
                'confidence': entity.get('score', 0.0)
            })
        
        return replaced_text, replacement_log
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """处理单个文本"""
        if not isinstance(text, str) or not text.strip():
            return {'text': text, 'entities': [], 'replaced': False}
        
        # 检测实体
        ner_entities = self.detect_entities(text)
        regex_entities = self.backup_regex_detection(text)
        
        # 合并实体
        all_entities = self.merge_entities(ner_entities, regex_entities)
        
        # 替换实体
        processed_text, replacement_log = self.replace_entities(text, all_entities)
        
        result = {
            'original_text': text,
            'processed_text': processed_text,
            'entities_detected': all_entities,
            'replacement_log': replacement_log,
            'replaced': len(replacement_log) > 0,
            'entities_count': len(all_entities)
        }
        
        if replacement_log:
            self.logger.info(f"在文本中检测到 {len(all_entities)} 个实体，进行了替换")
        
        return result
    
    def process_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """处理整个DataFrame"""
        result_df = df.copy()
        processing_stats = defaultdict(lambda: {'total_entities': 0, 'processed_texts': 0})
        
        for col in text_columns:
            self.logger.info(f"处理列: {col}")
            processed_col = []
            entity_counts = []
            
            for idx, text in enumerate(df[col]):
                if pd.isna(text):
                    processed_col.append(text)
                    entity_counts.append(0)
                    continue
                
                result = self.process_text(str(text))
                processed_col.append(result['processed_text'])
                entity_counts.append(result['entities_count'])
                
                processing_stats[col]['total_entities'] += result['entities_count']
                if result['replaced']:
                    processing_stats[col]['processed_texts'] += 1
                
                # 进度显示
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"已处理 {idx + 1}/{len(df)} 行")
            
            # 添加处理后的列和统计信息
            result_df[f'{col}_processed'] = processed_col
            result_df[f'{col}_entity_count'] = entity_counts
        
        # 保存处理统计
        stats_df = pd.DataFrame.from_dict(processing_stats, orient='index')
        stats_df.to_csv('ner_processing_stats.csv', encoding='utf-8-sig')
        
        self.logger.info("处理统计:")
        for col, stats in processing_stats.items():
            self.logger.info(f"{col}: 检测到 {stats['total_entities']} 个实体，处理了 {stats['processed_texts']} 个文本")
        
        return result_df
    
    def evaluate_detection(self, sample_texts: List[str]) -> pd.DataFrame:
        """评估实体检测效果"""
        evaluation_results = []
        
        for text in sample_texts:
            result = self.process_text(text)
            
            evaluation_results.append({
                'original_text': text,
                'processed_text': result['processed_text'],
                'entities_count': result['entities_count'],
                'entities_detected': [f"{e['entity_group']}:{e['word']}" for e in result['entities_detected']],
                'replacements_made': [f"{log['entity_type']}:{log['original']}->{log['replacement']}" 
                                    for log in result['replacement_log']]
            })
        
        return pd.DataFrame(evaluation_results)

class BiLSTMCRFMedicalNER:
    """基于BiLSTM-CRF的医学领域命名实体识别（简化版）"""
    
    def __init__(self, vocab_size=5000, embedding_dim=100, hidden_dim=128, num_tags=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 简化版的BiLSTM-CRF模型结构
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)
        
        # 标签映射（示例）
        self.tag2idx = {
            'O': 0, 'B-PER': 1, 'I-PER': 2,
            'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6,
            'B-MED': 7, 'I-MED': 8, 'B-DISEASE': 9
        }
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        
        self.model = nn.Sequential(
            self.embedding,
            self.lstm,
            self.hidden2tag
        ).to(self.device)
    
    def load_pretrained_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"加载预训练模型: {model_path}")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
    
    def predict(self, texts: List[str]) -> List[List[Dict]]:
        """预测文本中的实体（简化实现）"""
        # 这里应该是完整的预测逻辑，但需要训练好的模型
        # 这里返回示例结果
        results = []
        
        for text in texts:
            # 简化示例：使用规则匹配
            entities = []
            # 这里可以添加基于规则的实体识别作为后备方案
            
            results.append(entities)
        
        return results

def main():
    """主函数示例"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建NER检测器
    ner_detector = NERSensitiveInfoDetector()
    
    # 示例文本
    test_texts = [
        "患者张三丰，电话13812345678，在北京协和医院就诊，地址北京市朝阳区建国路100号。",
        "李四医生为患者王五开具了处方，联系方式li.si@hospital.com。",
        "患者身份证号码110101199001011234，银行卡号6222021234567890123。",
        "这是一段没有敏感信息的普通医疗咨询文本。"
    ]
    
    print("=== NER敏感信息检测和替换示例 ===")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n示例 {i}:")
        print(f"原始文本: {text}")
        
        result = ner_detector.process_text(text)
        print(f"处理后的文本: {result['processed_text']}")
        print(f"检测到的实体数量: {result['entities_count']}")
        
        if result['replacement_log']:
            print("替换记录:")
            for log in result['replacement_log']:
                print(f"  {log['entity_type']}: '{log['original']}' -> '{log['replacement']}'")
    
    # 评估检测效果
    print("\n=== 实体检测效果评估 ===")
    evaluation_df = ner_detector.evaluate_detection(test_texts)
    print(evaluation_df[['original_text', 'processed_text', 'entities_count']])
    
    # 处理CSV文件的示例
    try:
        df = pd.read_csv('medical_qa_data.csv', encoding='utf-8-sig')
        processed_df = ner_detector.process_dataframe(df, text_columns=['question_zh', 'answer_zh'])
        processed_df.to_csv('medical_qa_ner_processed.csv', index=False, encoding='utf-8-sig')
        print("\n数据文件处理完成！")
    except FileNotFoundError:
        print("\n未找到数据文件，跳过批量处理")

if __name__ == "__main__":
    main()