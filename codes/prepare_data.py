import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# --- 1. 配置区域 ---

# 输入文件：这是你用LLM标注好的、包含所有类别的完整文件
INPUT_FILE = "data/qwen_labeled_data_final.csv"

# 输出目录：存放最终划分好的三个文件的位置
OUTPUT_DIR = "data/final_split"

# 标签列的名称
LABEL_COLUMN = "qwen_label"

# 定义我们想要保留的六个核心类别
VALID_CATEGORIES = [
    "咨询",
    "求助",
    "投诉",
    "举报",
    "建议",
    "正面反馈"
]

# 数据集划分比例配置
TEST_SET_RATIO = 0.15  # 15% 作为测试集
VALIDATION_SET_RATIO = 0.15 # 15% 作为验证集
# 剩余 70% 将作为训练集

# 随机种子，保证每次划分结果一致
RANDOM_STATE = 42

# --- 2. 主执行逻辑 ---
if __name__ == "__main__":
    print("--- 开始执行数据准备流水线 ---")
    
    # 确保存放输出文件的目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 步骤 1/3: 加载原始标注数据 ---
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"步骤 1/3: 成功加载 '{INPUT_FILE}'，共 {len(df)} 条数据。")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_FILE}'。请确认文件路径和名称。")
        sys.exit(1)

    # --- 步骤 2/3: 筛选有效类别 ---
    original_rows = len(df)
    print(f"\n步骤 2/3: 开始筛选，只保留以下 {len(VALID_CATEGORIES)} 个核心类别:\n{VALID_CATEGORIES}")
    
    # 核心筛选逻辑
    df_filtered = df[df[LABEL_COLUMN].isin(VALID_CATEGORIES)].copy()
    # 重置索引，让数据更整洁
    df_filtered.reset_index(drop=True, inplace=True)
    
    num_removed = original_rows - len(df_filtered)
    print(f"筛选完成！共保留了 {len(df_filtered)} 条有效数据。")
    print(f"（移除了 {num_removed} 条'其他'类别或无效标签的数据）")

    if len(df_filtered) < 10:
        print("错误：筛选后的数据过少，无法进行划分。请检查数据质量或分类结果。")
        sys.exit(1)

    # --- 步骤 3/3: 划分训练、验证与测试集 ---
    print(f"\n步骤 3/3: 开始将 {len(df_filtered)} 条数据划分为训练/验证/测试集...")
    
    # 计算临时的测试集大小（验证集将在剩余数据中产生）
    temp_test_size = TEST_SET_RATIO + VALIDATION_SET_RATIO
    
    # 第一次划分：分出训练集和“剩余部分”（验证集+测试集）
    train_df, temp_df = train_test_split(
        df_filtered, 
        test_size=temp_test_size, 
        random_state=RANDOM_STATE,
        stratify=df_filtered[LABEL_COLUMN] # 分层抽样，保持类别分布
    )

    # 计算第二次划分的比例，以确保最终验证集和测试集大小正确
    final_test_size = VALIDATION_SET_RATIO / temp_test_size

    # 第二次划分：将“剩余部分”划分为验证集和测试集
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=final_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_df[LABEL_COLUMN]
    )

    # --- 保存最终文件 ---
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, encoding='utf-8-sig')
    validation_df.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False, encoding='utf-8-sig')
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, encoding='utf-8-sig')

    print("\n--- 数据准备流水线全部完成！ ---")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(validation_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"所有文件已成功保存至 '{OUTPUT_DIR}' 目录下。")