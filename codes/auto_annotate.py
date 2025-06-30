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
INPUT_CSV_PATH = "data/weibo_data_master.csv"
OUTPUT_CSV_PATH = "data/qwen_labeled_data.csv"
TEXT_COLUMN = "text"
NEW_LABEL_COLUMN = "qwen_label"
SAVE_INTERVAL = 50 

# --- 3. Prompt模板 ---
PROMPT_TEMPLATE = """你是一位逻辑严谨、经验丰富的城市交通领域舆情分析专家。你的任务是深入理解用户在社交媒体上发布的关于交通问题的真实意图，然后从【类别选项】中选择最恰当的一个作为输出。

在做决定时，请在“内心”里遵循以下思考路径：
1.  首先，分析文本的核心内容和潜在情感倾向（正面、负面、中性）。
2.  然后，逐一比对【类别选项】中的六个核心意图，判断文本是否主要符合其中某个类别的定义。
3.  最后，仅当文本不明确属于以上任何一个核心类别时，才将其归类为“其他”。

你的最终回答【必须】严格为【类别选项】中的一个词或词组，不要添加任何标点、数字、解释或多余的文字。

## 类别选项
* 咨询 (Consultation): 询问关于交通的客观信息，如政策、价格、路线、时刻等。
  例如：“请问从浦东机场到市区坐哪路公交最方便？”、“网约车开发票的流程是什么？”

* 求助 (Request for Help): 描述自己遇到的具体困难，请求帮助。
  例如：“我的交通卡掉在出租车上了，该联系谁？”、“这辆哈啰单车扫码后开不了锁，怎么办？”

* 投诉 (Complaint): 对服务、设施、管理等表达强烈不满，通常带有负面情绪。
  例如：“上海地铁的空调是打算把人冻死吗？”、“那个滴滴司机态度太差了，还故意绕路！”

* 举报 (Report): 揭发一个具体、正在发生的违规行为或安全隐患。
  例如：“举报这辆牌号为沪AX****的出租车不打表。”、“有人在公交车上抽烟，没人管。”

* 建议 (Suggestion): 提出具体的改进方案或想法。
  例如：“我建议你们在早高峰期间加密9号线的班次。”、“希望能在所有地铁站增加更多的休息座椅。”

* 正面反馈 (Positive Feedback): 对服务、人员、设施等表达明确的赞扬、感谢或满意。
  例如：“今天遇到的公交司机师傅超有耐心，为他点赞！”、“共享单车的调度快了很多，很方便。”

* 其他 (Other): 不属于以上任何类别。例如，纯粹的事实陈述、无明确意图的个人感想、闲聊或广告。
  例如：“今天路上好堵啊。”、“又是奔波在地铁上的一天。”

## 用户输入
用户文本: "{text}"

## 分类结果"""


# --- 4. API调用函数 ---

def get_label_from_llm(text_input, llm_client, model_name):
    """使用OpenAI兼容的客户端调用大模型获取【单个】分类标签。"""
    
    full_prompt = PROMPT_TEMPLATE.format(text=text_input)
    
    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0, # 温度设为0，追求最稳定、最符合指令的输出
        )
        
        label = response.choices[0].message.content.strip()
        return label

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