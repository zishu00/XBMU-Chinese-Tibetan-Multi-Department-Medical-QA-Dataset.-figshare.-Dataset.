import re
import pandas as pd
import json
from typing import List, Dict, Any
import logging

class RegexDataNormalizer:
    """使用正则表达式进行数据规范化和敏感信息替换"""
    
    def __init__(self):
        # 敏感信息正则模式
        self.sensitive_patterns = {
            'personal_name': [
                r'[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡魏贾丁薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文]某{1,2}',
                r'[A-Za-z]{2,20}\s?[A-Za-z]{2,20}'
            ],
            'phone_number': [
                r'1[3-9]\d{9}',
                r'\d{3,4}-\d{7,8}',
                r'\(\d{3,4}\)\d{7,8}'
            ],
            'id_card': [
                r'[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]',
                r'[1-9]\d{5}\d{8}[0-9Xx]'
            ],
            'medical_institution': [
                r'[北京上海天津重庆河北山西辽宁吉林黑龙江江苏浙江安徽福建江西山东河南湖北湖南广东海南四川贵州云南陕西甘肃青海台湾内蒙古广西西藏宁夏新疆][^，。！？]{1,10}医院',
                r'[^，。！？]{2,10}人民医院',
                r'[^，。！？]{2,10}中心医院',
                r'[^，。！？]{2,10}医科大学',
                r'[^，。！？]{2,10}卫生院',
                r'[^，。！？]{2,10}诊所'
            ],
            'address': [
                r'[省市区县街道路巷号栋单元室楼层]\d{1,5}号',
                r'\d{1,5}弄\d{1,5}号',
                r'[省市区县][^，。！？]{2,10}[市区县][^，。！？]{2,10}[街道路巷]'
            ],
            'email': [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            ],
            'bank_card': [
                r'\d{16,19}'
            ],
            'license_plate': [
                r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5}'
            ]
        }
        
        # 医学术语标准化映射
        self.medical_standardization = {
            # 疾病名称
            r'高血压病': '高血压',
            r'糖尿病病': '糖尿病',
            r'冠心疾病': '冠心病',
            r'脑卒中病': '脑卒中',
            r'胃溃疡病': '胃溃疡',
            r'支气管炎症': '支气管炎',
            r'肺炎症': '肺炎',
            r'肝炎病': '肝炎',
            
            # 症状描述
            r'头疼': '头痛',
            r'肚疼': '腹痛',
            r'拉肚子': '腹泻',
            r'发高烧': '发热',
            r'感冒病': '感冒',
            
            # 药物名称
            r'阿司匹林片': '阿司匹林',
            r'青霉素针': '青霉素',
            r'头孢类药物': '头孢类抗生素',
            
            # 单位标准化
            r'(\d+)\s*mg': r'\1毫克',
            r'(\d+)\s*g': r'\1克',
            r'(\d+)\s*kg': r'\1千克',
            r'(\d+)\s*ml': r'\1毫升',
            r'(\d+)\s*l': r'\1升',
            r'(\d+)\s*cm': r'\1厘米',
            r'(\d+)\s*m': r'\1米'
        }
        
        # 标点符号规范化
        self.punctuation_standardization = {
            r'，+': '，',
            r'。+': '。',
            r'！+': '！',
            r'？+': '？',
            r'；+': '；',
            r'：+': '：',
            r'\s+': ' ',
            r'\.{2,}': '。'
        }
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def replace_sensitive_info(self, text: str) -> str:
        """替换敏感信息为'某'"""
        if not isinstance(text, str) or not text.strip():
            return text
        
        original_text = text
        replaced_count = 0
        
        # 替换各种类型的敏感信息
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text)
                    if matches:
                        # 根据类别选择替换策略
                        if category in ['personal_name']:
                            replacement = '某某'
                        elif category in ['phone_number', 'id_card', 'bank_card']:
                            replacement = '某' * 4
                        else:
                            replacement = '某'
                        
                        # 执行替换
                        text = re.sub(pattern, replacement, text)
                        replaced_count += len(matches)
                        
                        if matches:
                            self.logger.debug(f"在文本中发现{category}: {matches} -> 替换为: {replacement}")
                            
                except re.error as e:
                    self.logger.warning(f"正则表达式错误 {pattern}: {e}")
                    continue
        
        if replaced_count > 0:
            self.logger.info(f"替换了 {replaced_count} 处敏感信息")
        
        return text
    
    def standardize_medical_terms(self, text: str) -> str:
        """标准化医学术语"""
        if not isinstance(text, str):
            return text
        
        for pattern, replacement in self.medical_standardization.items():
            try:
                text = re.sub(pattern, replacement, text)
            except re.error as e:
                self.logger.warning(f"医学术语标准化错误 {pattern}: {e}")
                continue
        
        return text
    
    def standardize_punctuation(self, text: str) -> str:
        """标准化标点符号"""
        if not isinstance(text, str):
            return text
        
        for pattern, replacement in self.punctuation_standardization.items():
            try:
                text = re.sub(pattern, replacement, text)
            except re.error as e:
                self.logger.warning(f"标点符号标准化错误 {pattern}: {e}")
                continue
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """完整的文本规范化流程"""
        if not isinstance(text, str) or not text.strip():
            return text
        
        # 执行规范化步骤
        text = self.replace_sensitive_info(text)
        text = self.standardize_medical_terms(text)
        text = self.standardize_punctuation(text)
        
        return text
    
    def process_dataframe(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """处理整个DataFrame"""
        if text_columns is None:
            # 自动检测文本列
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object' and any(isinstance(x, str) for x in df[col].dropna()[:10]):
                    text_columns.append(col)
        
        self.logger.info(f"将对以下列进行规范化处理: {text_columns}")
        
        result_df = df.copy()
        stats = {}
        
        for col in text_columns:
            self.logger.info(f"处理列: {col}")
            original_samples = result_df[col].dropna()
            total_samples = len(original_samples)
            
            if total_samples == 0:
                continue
            
            # 应用规范化
            result_df[col] = result_df[col].apply(self.normalize_text)
            
            # 统计信息
            modified_count = 0
            for orig, new in zip(original_samples, result_df[col].dropna()):
                if orig != new:
                    modified_count += 1
            
            stats[col] = {
                'total_samples': total_samples,
                'modified_samples': modified_count,
                'modification_rate': modified_count / total_samples if total_samples > 0 else 0
            }
            
            self.logger.info(f"列 {col}: 修改了 {modified_count}/{total_samples} 个样本 ({stats[col]['modification_rate']:.2%})")
        
        # 保存统计信息
        with open('normalization_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return result_df
    
    def validate_normalization(self, original_text: str, normalized_text: str) -> Dict[str, Any]:
        """验证规范化效果"""
        validation_result = {
            'original_length': len(original_text),
            'normalized_length': len(normalized_text),
            'changed': original_text != normalized_text,
            'sensitive_info_removed': False,
            'changes': []
        }
        
        if original_text != normalized_text:
            # 简单的变化检测（实际可以使用更复杂的diff算法）
            if len(original_text) != len(normalized_text):
                validation_result['changes'].append('文本长度发生变化')
            
            # 检查是否移除了敏感信息
            sensitive_detected = False
            for category, patterns in self.sensitive_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, original_text) and not re.search(pattern, normalized_text):
                        sensitive_detected = True
                        validation_result['changes'].append(f'移除了{category}')
            
            validation_result['sensitive_info_removed'] = sensitive_detected
        
        return validation_result

def main():
    """主函数示例"""
    # 创建规范化器
    normalizer = RegexDataNormalizer()
    
    # 示例文本
    test_texts = [
        "患者张三，电话13812345678，身份证110101199001011234，在北京协和医院就诊。",
        "李四医生开具了阿司匹林片100mg，每日一次。",
        "地址：北京市朝阳区建国路100号，邮箱zhangsan@example.com。",
        "患者反映头痛、肚疼、发高烧等症状。"
    ]
    
    print("=== 正则表达式规范化示例 ===")
    for i, text in enumerate(test_texts, 1):
        print(f"\n示例 {i}:")
        print(f"原始: {text}")
        normalized = normalizer.normalize_text(text)
        print(f"规范后: {normalized}")
        
        # 验证
        validation = normalizer.validate_normalization(text, normalized)
        print(f"验证: {validation}")
    
    # 处理CSV文件的示例
    try:
        df = pd.read_csv('medical_qa_data.csv', encoding='utf-8-sig')
        normalized_df = normalizer.process_dataframe(df, text_columns=['question_zh', 'answer_zh'])
        normalized_df.to_csv('medical_qa_normalized.csv', index=False, encoding='utf-8-sig')
        print("\n数据文件处理完成！")
    except FileNotFoundError:
        print("\n未找到数据文件，跳过批量处理")

if __name__ == "__main__":
    main()