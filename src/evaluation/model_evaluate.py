from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, Features, ClassLabel, Value
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- 1. 配置区域 ---
# MODEL_CHECKPOINT_PATH = "./roberta_hyper_search_results/run-18/checkpoint-537"
MODEL_CHECKPOINT_PATH = "./models/mac_hyper_search_results/run-11/checkpoint-716"
DATA_DIR = "output/processed_data/final_split/"
TEXT_COLUMN = "text"
ORIGINAL_LABEL_COLUMN = "qwen_label" # <-- 【修正点】使用CSV文件中真实的列名
OUTPUT_DIR = "./src/evaluation_results/mac_best_model"

# --- 2. 主逻辑 ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"错误：找不到模型路径 '{MODEL_CHECKPOINT_PATH}'。请确认训练已完成，并检查路径是否指向具体的checkpoint文件夹或最终模型目录。")
        sys.exit(1)

    print("--- 开始加载模型和测试数据 ---")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)
    
    # --- !!! 核心修正 !!! ---
    # 1. 从模型配置中获取标签列表
    label_names = list(model.config.id2label.values())
    print(f"从模型配置中读取到标签列表: {label_names}")
    
    # 2. 构建与CSV文件完全匹配的Features对象
    features = Features({
        'post_id': Value('string'),
        'created_at': Value('string'),
        'user_name': Value('string'),
        TEXT_COLUMN: Value('string'),
        ORIGINAL_LABEL_COLUMN: ClassLabel(names=label_names) # <-- 使用真实的列名 'qwen_label'
    })

    # 3. 使用正确的Features加载测试集
    test_dataset = load_dataset(
        'csv', 
        data_files={'test': os.path.join(DATA_DIR, 'test.csv')},
        features=features
    )['test']

    # 4. 加载成功后，再将列名重命名为'label'，以符合Trainer的内部期望
    test_dataset_renamed = test_dataset.rename_column(ORIGINAL_LABEL_COLUMN, "label")
    print("已将标签列重命名为 'label' 以进行评估。")
    # ---------------------------------------------
    
    # 对测试集进行预处理
    def preprocess_function(examples):
        return tokenizer(examples[TEXT_COLUMN], truncation=True, padding=True, max_length=256)
        
    tokenized_test_dataset = test_dataset_renamed.map(preprocess_function, batched=True, remove_columns=[TEXT_COLUMN])
    
    trainer = Trainer(model=model)
    
    print("\n--- 正在对测试集进行预测 ---")
    
    predictions_output = trainer.predict(tokenized_test_dataset)
    logits = predictions_output.predictions
    y_true = predictions_output.label_ids
    y_pred = np.argmax(logits, axis=1)

    # --- 3. 生成并打印分类报告 ---
    print("\n--- 详细分类报告 ---")
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    print(report)
    
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"分类报告已保存至 '{report_path}'")

    # --- 4. 生成并绘制混淆矩阵 ---
    print("\n--- 正在生成混淆矩阵 ---")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
    plt.ylabel('真实标签 (True Label)', fontsize=12)
    plt.title('混淆矩阵 (Confusion Matrix)', fontsize=14)
    
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图片已保存至 '{cm_path}'")
    
    print("\n评估完成！")