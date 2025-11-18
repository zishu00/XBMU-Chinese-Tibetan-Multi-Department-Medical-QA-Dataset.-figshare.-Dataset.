import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from tqdm import tqdm
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('roberta_classifier.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalQADataset(Dataset):
    """医疗问答数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RoBERTaMedicalClassifier:
    """基于RoBERTa的医疗科室分类器"""
    
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', num_labels=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {self.device}')
        
        # 加载tokenizer和模型
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # 科室映射（根据论文中的六大科室）
        self.department_mapping = {
            '耳鼻喉科': 0,
            '眼科': 1,
            '内科': 2,
            '神经内科': 3,
            '外科': 4,
            '营养保健科': 5
        }
        
        self.reverse_mapping = {v: k for k, v in self.department_mapping.items()}
        
        # 训练参数
        self.best_accuracy = 0
        self.training_history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
    
    def preprocess_text(self, text):
        """预处理文本"""
        if pd.isna(text):
            return ""
        
        # 清理文本
        text = re.sub(r'\s+', ' ', str(text))
        text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\w\s\.\,\?\!]', '', text)
        return text.strip()
    
    def prepare_data(self, df, text_column='question_zh', label_column='category'):
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        # 过滤有效数据
        valid_data = df[
            (df[text_column].notna()) & 
            (df[label_column].notna()) &
            (df[label_column].isin(self.department_mapping.keys()))
        ].copy()
        
        logger.info(f"有效数据量: {len(valid_data)}")
        
        # 预处理文本
        valid_data['processed_text'] = valid_data[text_column].apply(self.preprocess_text)
        
        # 过滤空文本
        valid_data = valid_data[valid_data['processed_text'].str.len() > 0]
        
        # 转换为数值标签
        valid_data['label'] = valid_data[label_column].map(self.department_mapping)
        
        # 统计科室分布
        department_counts = valid_data[label_column].value_counts()
        logger.info("科室分布:")
        for dept, count in department_counts.items():
            logger.info(f"  {dept}: {count}")
        
        texts = valid_data['processed_text'].tolist()
        labels = valid_data['label'].tolist()
        
        return texts, labels
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              epochs=5, batch_size=16, learning_rate=2e-5, validation_split=0.2):
        """训练模型"""
        
        # 划分训练集和验证集
        if val_texts is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, 
                test_size=validation_split, 
                random_state=42,
                stratify=train_labels
            )
        
        logger.info(f"训练集大小: {len(train_texts)}")
        logger.info(f"验证集大小: {len(val_texts)}")
        
        # 创建数据集
        train_dataset = MedicalQADataset(train_texts, train_labels, self.tokenizer)
        val_dataset = MedicalQADataset(val_texts, val_labels, self.tokenizer)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器和调度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 训练循环
        for epoch in range(epochs):
            logger.info(f'开始第 {epoch+1}/{epochs} 轮训练')
            
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch+1}')
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            val_accuracy, avg_val_loss = self.evaluate(val_loader)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['val_loss'].append(avg_val_loss)
            
            logger.info(f'训练损失: {avg_train_loss:.4f}, 验证准确率: {val_accuracy:.4f}, 验证损失: {avg_val_loss:.4f}')
            
            # 保存最佳模型
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.save_model('best_roberta_medical_classifier.pth')
                logger.info(f'新的最佳模型已保存，准确率: {val_accuracy:.4f}')
        
        # 加载最佳模型
        self.load_model('best_roberta_medical_classifier.pth')
        logger.info(f'训练完成，最佳验证准确率: {self.best_accuracy:.4f}')
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        total_eval_samples = 0
        
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_eval_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions = (predictions == labels).sum().item()
                
                total_eval_accuracy += correct_predictions
                total_eval_samples += len(labels)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
        
        accuracy = total_eval_accuracy / total_eval_samples
        avg_loss = total_eval_loss / len(dataloader)
        
        return accuracy, avg_loss
    
    def predict(self, texts, batch_size=16, confidence_threshold=0.6):
        """预测科室类别"""
        self.model.eval()
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 创建数据集
        dummy_labels = [0] * len(processed_texts)  # 占位标签
        dataset = MedicalQADataset(processed_texts, dummy_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_confidences = []
        all_need_review = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='预测'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # 转换为科室名称
        predicted_departments = [self.reverse_mapping[pred] for pred in all_predictions]
        
        # 标记需要人工复核的样本
        for conf in all_confidences:
            all_need_review.append(conf < confidence_threshold)
        
        results = []
        for i, (dept, conf, need_review) in enumerate(zip(predicted_departments, all_confidences, all_need_review)):
            results.append({
                'text': texts[i],
                'predicted_department': dept,
                'confidence': float(conf),
                'need_human_review': bool(need_review)
            })
        
        return results
    
    def predict_dataframe(self, df, text_column='question_zh', confidence_threshold=0.6):
        """对DataFrame进行预测"""
        texts = df[text_column].tolist()
        predictions = self.predict(texts, confidence_threshold=confidence_threshold)
        
        # 将预测结果添加到DataFrame
        result_df = df.copy()
        result_df['predicted_category'] = [p['predicted_department'] for p in predictions]
        result_df['prediction_confidence'] = [p['confidence'] for p in predictions]
        result_df['need_human_review'] = [p['need_human_review'] for p in predictions]
        
        # 统计需要人工复核的比例
        need_review_count = result_df['need_human_review'].sum()
        total_count = len(result_df)
        logger.info(f"需要人工复核的样本: {need_review_count}/{total_count} ({need_review_count/total_count*100:.2f}%)")
        
        return result_df
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'department_mapping': self.department_mapping,
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }, filepath)
        logger.info(f'模型已保存到: {filepath}')
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.department_mapping = checkpoint['department_mapping']
        self.reverse_mapping = {v: k for k, v in self.department_mapping.items()}
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
        logger.info(f'模型已从 {filepath} 加载')
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失
        ax1.plot(self.training_history['train_loss'], label='训练损失')
        ax1.plot(self.training_history['val_loss'], label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 绘制准确率
        ax2.plot(self.training_history['val_accuracy'], label='验证准确率', color='orange')
        ax2.set_title('验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    
    # 1. 加载数据
    logger.info("加载数据...")
    try:
        # 假设数据保存在CSV文件中
        df = pd.read_csv('dxy_medical_qa_cleaned.csv', encoding='utf-8-sig')
        logger.info(f"数据加载成功，共 {len(df)} 条记录")
    except FileNotFoundError:
        logger.error("数据文件未找到，请先运行爬虫代码获取数据")
        return
    
    # 检查必要的列
    required_columns = ['question_zh', 'category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"缺少必要的列: {missing_columns}")
        return
    
    # 2. 初始化分类器
    classifier = RoBERTaMedicalClassifier()
    
    # 3. 准备数据
    texts, labels = classifier.prepare_data(df)
    
    # 4. 训练模型
    logger.info("开始训练RoBERTa分类器...")
    classifier.train(
        texts, 
        labels, 
        epochs=5, 
        batch_size=16, 
        learning_rate=2e-5,
        validation_split=0.2
    )
    
    # 5. 绘制训练历史
    classifier.plot_training_history()
    
    # 6. 对整个数据集进行预测（包含置信度筛选）
    logger.info("对整个数据集进行预测...")
    result_df = classifier.predict_dataframe(df, confidence_threshold=0.6)
    
    # 7. 保存结果
    result_df.to_csv('medical_qa_with_predictions.csv', index=False, encoding='utf-8-sig')
    logger.info("预测结果已保存到 medical_qa_with_predictions.csv")
    
    # 8. 分离需要人工复核的数据
    need_review_df = result_df[result_df['need_human_review']]
    if len(need_review_df) > 0:
        need_review_df.to_csv('need_human_review_samples.csv', index=False, encoding='utf-8-sig')
        logger.info(f"需要人工复核的样本已保存到 need_human_review_samples.csv")
    
    # 9. 统计预测结果
    department_distribution = result_df['predicted_category'].value_counts()
    logger.info("预测科室分布:")
    for dept, count in department_distribution.items():
        logger.info(f"  {dept}: {count}")
    
    # 如果有真实标签，计算准确率
    if 'category' in result_df.columns:
        accuracy = accuracy_score(result_df['category'], result_df['predicted_category'])
        logger.info(f"整体准确率: {accuracy:.4f}")
        
        # 详细分类报告
        report = classification_report(
            result_df['category'], 
            result_df['predicted_category'],
            target_names=classifier.department_mapping.keys()
        )
        logger.info("详细分类报告:\n" + report)
        
        # 混淆矩阵
        cm = confusion_matrix(result_df['category'], result_df['predicted_category'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classifier.department_mapping.keys(),
                   yticklabels=classifier.department_mapping.keys())
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()