import pandas as pd
import time
import random
import os
import sys
from tqdm import tqdm
from openai import OpenAI
import numpy as np

# --- 1. 从系统环境变量中加载配置 ---
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    print("错误：未能从系统环境变量中读取到 'DASHSCOPE_API_KEY'。")
    print("\n请确认：")
    print("1. 你已经成功设置了系统环境变量。")
    print("2. （非常重要）你已经重启了当前这个终端窗口。")
    sys.exit(1)

# --- 2. 配置区域 ---
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
LLM_MODEL_NAME = "qwen-turbo"

# --- 文件与列名配置 ---
INPUT_CSV_PATH = "data/weibo_data_master.csv"
FINAL_OUTPUT_CSV_PATH = "output/processed_data/qwen_labeled_data_final.csv"
CHUNK_OUTPUT_DIR = "output/processed_data/labeled_chunks" # 存放中间分块文件的目录

TEXT_COLUMN = "text"
POST_ID_COLUMN = "post_id" # 用于去重的列
NEW_LABEL_COLUMN = "qwen_label"

# --- 任务配置 ---
NUM_CHUNKS = 5 # 要分成的块数
SAVE_INTERVAL = 50 

# --- 3. Prompt模板 (保持不变) ---
PROMPT_TEMPLATE = """你是一位逻辑严谨、经验丰富的城市交通领域舆情分析专家。你的任务是深入理解用户在社交媒体上发布的关于交通问题的真实意图，然后从【类别选项】中选择最恰当的一个作为输出。

在做决定时，请在“内心”里遵循以下思考路径：
1.  首先，分析文本的核心内容和潜在情感倾向（正面、负面、中性）。
2.  然后，逐一比对【类别选项】中的六个核心意图，判断文本是否主要符合其中某个类别的定义。
3.  最后，仅当文本不明确属于以上任何一个核心类别时，才将其归类为“其他”。

你的最终回答【必须】严格为【类别选项】中的一个词或词组，不要添加任何标点、数字、解释或多余的文字。

## 类别选项
* 咨询 (Consultation): 询问关于交通的客观信息，如政策、价格、路线、时刻等。
* 求助 (Request for Help): 描述自己遇到的具体困难，请求帮助。
* 投诉 (Complaint): 对服务、设施、管理等表达强烈不满，通常带有负面情绪。
* 举报 (Report): 揭发一个具体、正在发生的违规行为或安全隐患。
* 建议 (Suggestion): 提出具体的改进方案或想法。
* 正面反馈 (Positive Feedback): 对服务、人员、设施等表达明确的赞扬、感谢或满意。
* 其他 (Other): 不属于以上任何类别。例如，纯粹的事实陈述、无明确意图的个人感想、闲聊或广告。

## 用户输入
用户文本: "{text}"

## 分类结果"""


# --- 4. API调用函数 (保持不变) ---
def get_label_from_llm(text_input, llm_client, model_name):
    """使用OpenAI兼容的客户端调用大模型获取【单个】分类标签。"""
    full_prompt = PROMPT_TEMPLATE.format(text=text_input)
    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nAPI调用时发生错误: {e}")
        return "API_ERROR"

# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    # --- 步骤一：加载、去重、分块 ---
    print("--- 步骤 1/3: 加载、去重并分块数据 ---")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"成功加载输入文件，共 {len(df)} 条原始数据。")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_CSV_PATH}'。")
        sys.exit(1)

    original_rows = len(df)
    df.drop_duplicates(subset=[POST_ID_COLUMN], keep='last', inplace=True)
    print(f"去重完成，剩余 {len(df)} 条唯一数据。共移除了 {original_rows - len(df)} 条重复项。")

    # 将数据分成N块
    chunks = np.array_split(df, NUM_CHUNKS)
    print(f"数据已成功均分为 {len(chunks)} 块，每块约 {len(chunks[0])} 条数据。")

    # 确保存放分块文件的目录存在
    os.makedirs(CHUNK_OUTPUT_DIR, exist_ok=True)

    # --- 步骤二：分批处理每个数据块 ---
    print("\n--- 步骤 2/3: 开始分批处理数据块 ---")
    for i, chunk_df in enumerate(chunks):
        chunk_filename = os.path.join(CHUNK_OUTPUT_DIR, f"labeled_chunk_{i+1}.csv")
        
        # 检查是否已处理过此块，实现断点续处理
        if os.path.exists(chunk_filename):
            print(f"检测到 {chunk_filename} 已存在，跳过第 {i+1}/{NUM_CHUNKS} 块的处理。")
            continue

        print(f"\n--- 正在处理第 {i+1}/{NUM_CHUNKS} 块数据 ---")
        
        # 为当前块创建一个新列用于存放标签
        chunk_df_copy = chunk_df.copy()
        chunk_df_copy[NEW_LABEL_COLUMN] = ""

        # 对当前块的每一行进行标注
        for index in tqdm(chunk_df_copy.index, desc=f"Annotating Chunk {i+1}"):
            text_to_label = chunk_df_copy.loc[index, TEXT_COLUMN]
            if not isinstance(text_to_label, str) or len(text_to_label.strip()) < 5:
                chunk_df_copy.loc[index, NEW_LABEL_COLUMN] = "文本格式错误"
                continue
            
            predicted_label = get_label_from_llm(text_to_label, client, LLM_MODEL_NAME)
            chunk_df_copy.loc[index, NEW_LABEL_COLUMN] = predicted_label
            
            time.sleep(random.uniform(0.5, 1.5))

        # 保存当前处理好的数据块
        chunk_df_copy.to_csv(chunk_filename, index=False, encoding='utf-8-sig')
        print(f"第 {i+1}/{NUM_CHUNKS} 块处理完成，结果已保存至 {chunk_filename}")

    # --- 步骤三：合并所有已处理的数据块 ---
    print("\n--- 步骤 3/3: 合并所有数据块 ---")
    all_labeled_chunks = []
    all_chunks_processed = True
    for i in range(NUM_CHUNKS):
        chunk_filename = os.path.join(CHUNK_OUTPUT_DIR, f"labeled_chunk_{i+1}.csv")
        if os.path.exists(chunk_filename):
            all_labeled_chunks.append(pd.read_csv(chunk_filename))
        else:
            print(f"警告：找不到分块文件 {chunk_filename}，最终合并结果将不完整。")
            all_chunks_processed = False
    
    if all_labeled_chunks:
        final_df = pd.concat(all_labeled_chunks, ignore_index=True)
        final_df.to_csv(FINAL_OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        if all_chunks_processed:
            print("\n--- 所有任务成功完成！---")
        else:
            print("\n--- 部分任务完成 ---")
        print(f"最终合并的数据集已保存至: {FINAL_OUTPUT_CSV_PATH}，共包含 {len(final_df)} 条数据。")
    else:
        print("错误：未能找到任何已处理的数据块，无法生成最终文件。")