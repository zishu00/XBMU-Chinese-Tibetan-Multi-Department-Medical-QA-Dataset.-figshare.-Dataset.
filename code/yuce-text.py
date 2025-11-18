def process_new_data_with_classifier(input_file, output_file, model_path='best_roberta_medical_classifier.pth'):
    """使用训练好的分类器处理新数据"""
    
    # 加载新数据
    logger.info(f"加载新数据: {input_file}")
    new_df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    # 初始化分类器并加载训练好的模型
    classifier = RoBERTaMedicalClassifier()
    classifier.load_model(model_path)
    
    # 进行预测
    logger.info("开始预测科室类别...")
    result_df = classifier.predict_dataframe(new_df, confidence_threshold=0.6)
    
    # 保存结果
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"预测结果已保存到: {output_file}")
    
    # 输出统计信息
    total_count = len(result_df)
    need_review_count = result_df['need_human_review'].sum()
    high_confidence_count = total_count - need_review_count
    
    logger.info(f"总计: {total_count} 条数据")
    logger.info(f"高置信度预测: {high_confidence_count} 条 ({high_confidence_count/total_count*100:.2f}%)")
    logger.info(f"需要人工复核: {need_review_count} 条 ({need_review_count/total_count*100:.2f}%)")
    
    # 科室分布
    department_dist = result_df['predicted_category'].value_counts()
    logger.info("预测科室分布:")
    for dept, count in department_dist.items():
        percentage = count / total_count * 100
        logger.info(f"  {dept}: {count} 条 ({percentage:.2f}%)")
    
    return result_df

# 使用示例
if __name__ == "__main__":
    # 处理新数据
    result_df = process_new_data_with_classifier(
        input_file='new_medical_questions.csv',
        output_file='new_questions_with_departments.csv'
    )