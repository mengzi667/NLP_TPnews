import pandas as pd
from datasets import load_dataset, Features, ClassLabel, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import f1_score
import optuna
import os
import sys

# --- 1. 配置区域 ---
MODEL_NAME = "hfl/chinese-macbert-base"
DATA_DIR = "data/final_split/"
OUTPUT_DIR_BASE = "./models/mac_hyper_search_results"
ORIGINAL_LABEL_COLUMN = "qwen_label"
FINAL_LABEL_COLUMN = "label"
TEXT_COLUMN = "text"

# --- 2. 数据加载与预处理 ---
def load_and_prepare_data():
    train_csv_path = os.path.join(DATA_DIR, 'train.csv')
    if not os.path.exists(train_csv_path):
        print(f"错误：找不到训练文件 {train_csv_path}。请先运行 prepare_dataset.py 脚本。")
        sys.exit(1)
        
    train_df = pd.read_csv(train_csv_path, encoding="utf-8-sig")
    labels_list = sorted(train_df[ORIGINAL_LABEL_COLUMN].unique().tolist())
    
    features = Features({
        'post_id': Value('string'), 'created_at': Value('string'),
        'user_name': Value('string'), TEXT_COLUMN: Value('string'),
        ORIGINAL_LABEL_COLUMN: ClassLabel(names=labels_list)
    })
    dataset = load_dataset('csv', data_files={
        'train': train_csv_path,
        'validation': os.path.join(DATA_DIR, 'validation.csv')
    }, features=features)
    dataset = dataset.rename_column(ORIGINAL_LABEL_COLUMN, FINAL_LABEL_COLUMN)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    

    def preprocess_function(examples):
        return tokenizer(examples[TEXT_COLUMN], truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=[TEXT_COLUMN, 'post_id', 'created_at', 'user_name'])
    
    return tokenized_dataset, labels_list, tokenizer

def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )

# --- 定义评估指标和超参数空间 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='macro')
    return {'f1_macro': f1}

def my_hp_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 5e-5]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
    }

# --- 主执行逻辑 ---
if __name__ == "__main__":
    tokenized_data, labels, tokenizer = load_and_prepare_data()
    NUM_LABELS = len(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_BASE,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n--- 开始自动化超参数搜索 ---")
    
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=my_hp_space,
        n_trials=20,
    )

    print("\n--- 超参数搜索完成！ ---")
    print(f"最佳试验结果 (Best Trial):")
    print(best_trial)