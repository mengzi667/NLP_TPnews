import pandas as pd
import time
import random
import os
import sys
from tqdm import tqdm
from openai import OpenAI

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
INPUT_CSV_PATH = "上海地铁_data.csv" # TODO: 确认这是你采集到的数据文件名
OUTPUT_CSV_PATH = "qwen_labeled_data_v2.csv" # 建议用新名字以区分版本
TEXT_COLUMN = "text"
NEW_LABEL_COLUMN = "qwen_label"
SAVE_INTERVAL = 50 

# --- 3. Prompt模板 ---
PROMPT_TEMPLATE = """你是一位精通政务服务和城市管理的意图分类专家。你的任务是先对用户输入的文本进行一步步的分析，理解其核心意图，然后再将其严格归类到以下六个预设类别中。你的回答必须只能是这些类别中的一个，不要添加任何解释或多余的文字。

类别选项:
- 咨询 (Consultation): 公民对政策、办事流程等信息的询问。例如：“请问如何办理公交卡？”、“末班车是几点？”
- 求助 (Request for Help): 公民在生活中遇到困难或意外，请求提供非紧急救助。例如：“我的钱包好像掉在车上了，怎么办？”、“小孩走丢了，谁能帮帮我。”
- 投诉 (Complaint): 公民对管理、服务、产品等方面表达强烈不满。例如：“你们这个安检效率也太低了！搞什么呢？”、“上海地铁空调能把人冻死，完全不考虑乘客感受！”
- 举报 (Report): 公民向有关部门揭发违法违规行为或安全隐患。例如：“有人在车厢里抽烟，没人管。”、“我看到有人翻越闸机逃票。”
- 建议 (Suggestion): 公民对城市管理、社会治理等方面提出具体的改进方案。例如：“建议早高峰期间增开几班车。”、“希望延长11号线的夜间运营时间。”
- 其他 (Other): 不属于以上任何类别的情绪表达、闲聊、广告或无明确意图的文本。例如：“今天天气真好。”、“上班打卡。”

用户文本: "{text}"

分类结果:"""


# --- 4. API调用函数 ---
def get_label_from_llm(text_input, llm_client, model_name):
    """使用OpenAI兼容的客户端调用大模型获取标签。"""
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
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_CSV_PATH}'。")
        sys.exit(1)

    start_index = 0
    if os.path.exists(OUTPUT_CSV_PATH):
        df_labeled = pd.read_csv(OUTPUT_CSV_PATH)
        start_index = len(df_labeled)
        print(f"检测到已存在的输出文件，将从第 {start_index + 1} 条数据开始继续标注。")
        df.loc[:start_index-1, NEW_LABEL_COLUMN] = df_labeled[NEW_LABEL_COLUMN]
    else:
        df[NEW_LABEL_COLUMN] = ""
        print("未检测到输出文件，将从头开始标注。")

    print(f"--- 开始使用通义千问({LLM_MODEL_NAME})进行自动化标注 ---")
    for index in tqdm(range(start_index, len(df)), initial=start_index, total=len(df), desc="Annotating"):
        text_to_label = df.loc[index, TEXT_COLUMN]
        if not isinstance(text_to_label, str) or len(text_to_label.strip()) < 5:
            df.loc[index, NEW_LABEL_COLUMN] = "文本格式错误"
            continue
        
        predicted_label = get_label_from_llm(text_to_label, client, LLM_MODEL_NAME)
        df.loc[index, NEW_LABEL_COLUMN] = predicted_label
        
        time.sleep(random.uniform(0.5, 1.5))

        if (index + 1) % SAVE_INTERVAL == 0:
            df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print("\n--- 自动化标注任务全部完成！---")
    print(f"结果已保存至: {OUTPUT_CSV_PATH}")