import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import os

# --- 1. 页面配置 (Page Configuration) ---
# st.set_page_config() 必须是脚本中第一个被调用的Streamlit命令
st.set_page_config(
    page_title="城市交通舆情意图识别系统",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. 加载模型 (Model Loading) ---
# 使用 @st.cache_resource 装饰器来缓存模型，避免每次刷新页面都重新加载
# 这会极大地提升应用的响应速度
@st.cache_resource
def load_model():
    """
    加载本地训练好的模型和分词器。
    这个函数只会在第一次运行时执行，之后的结果会被缓存。
    """
    # TODO: (非常重要!) 将这里的路径替换为你最终的、性能最好的那个模型checkpoint的路径
    # 例如: "./model_results_weighted/checkpoint-1611"
    model_path = "./models/mac_hyper_search_results/run-11/checkpoint-716" 

    try:
        # 检查路径是否存在
        if not os.path.exists(model_path):
            st.error(f"错误：找不到模型路径 '{model_path}'。请确认路径是否正确。")
            return None, None
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("模型和分词器加载成功！")
        return tokenizer, model
    except Exception as e:
        st.error(f"模型加载失败，请检查路径是否为有效的Hugging Face模型目录。")
        st.error(f"错误详情: {e}")
        return None, None

# 执行加载
tokenizer, model = load_model()

# --- 3. 页面标题和简介 (Title and Introduction) ---
st.title("🚇 城市交通舆情意图识别系统")
st.markdown("""
欢迎使用本系统！这是一个基于 `MacBERT` 微调模型的自然语言处理应用，能够自动识别关于**城市交通**（地铁、公交、出租车等）的文本所表达的核心意图。
您可以在下方的文本框中输入任何反馈、问题或建议，模型将会给出它的判断。
""")
st.markdown("---")


# --- 4. 主应用界面 (Main Interface) ---
if model is None:
    st.warning("模型未能成功加载，应用无法正常工作。请检查终端中的错误信息。")
else:
    col1, col2 = st.columns([2, 1]) # 创建两列，左边更宽

    with col1:
        st.header("✏️ 请输入要分析的文本:")
        # 使用表单来组织输入和按钮，可以防止每次输入都重新运行整个页面
        with st.form("intent_form"):
            text_input = st.text_area(
                "输入文本:", 
                "强烈建议地铁11号线在早高峰增加几班车，现在等一趟的时间也太长了！", 
                height=200,
                placeholder="例如：上海地铁的空调是打算把人冻死吗？"
            )
            submitted = st.form_submit_button("🚀 开始分析")

    with col2:
        st.header("💡 意图类别说明")
        st.info("""
        - **咨询:** 询问信息、流程等。
        - **求助:** 请求具体帮助（如寻物）。
        - **投诉:** 表达强烈不满。
        - **举报:** 揭发违规或安全隐患。
        - **建议:** 提出具体改进方案。
        - **正面反馈:** 明确的赞扬或感谢。
        """)

    # --- 5. 模型推理与结果展示 (Inference and Display) ---
    if submitted:
        if not text_input.strip():
            st.warning("请输入有效的文本内容后再进行分析。")
        else:
            with st.spinner('模型正在全力分析中，请稍候...'):
                # a. 预处理输入
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                
                # b. 模型推理
                with torch.no_grad(): # 在推理模式下，不计算梯度以加速
                    outputs = model(**inputs)
                
                # c. 后处理结果
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1).flatten().tolist()
                
                # 获取标签名称和对应的概率
                labels = list(model.config.id2label.values())
                prob_df = pd.DataFrame({
                    '意图类别': labels,
                    '概率': probabilities
                }).sort_values(by='概率', ascending=False).reset_index(drop=True)

                # d. 展示结果
                st.markdown("---")
                st.header("📈 分析结果")
                
                # 分两列展示
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    # 使用st.metric展示最可能的类别
                    st.metric(
                        label="**最高概率意图**", 
                        value=prob_df.iloc[0]['意图类别'],
                        help=f"模型认为这条文本最有可能属于这个类别，置信度为 {prob_df.iloc[0]['概率']:.2%}"
                    )
                    st.write("详细概率分布：")
                    # 使用st.dataframe展示带进度条的表格
                    st.dataframe(prob_df,
                                  use_container_width=True,
                                  column_config={
                                      "概率": st.column_config.ProgressColumn(
                                          "概率",
                                          format="%.2f%%",
                                          min_value=0,
                                          max_value=1,
                                      ),
                                  },
                                  hide_index=True)

                with res_col2:
                    # 使用Plotly绘制漂亮的条形图
                    fig = px.bar(
                        prob_df, 
                        x='概率', 
                        y='意图类别', 
                        orientation='h', 
                        title='各意图类别概率分布图',
                        text_auto='.2%',
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        xaxis_title="概率",
                        yaxis_title="意图类别"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# --- 6. 侧边栏 (Sidebar) ---
st.sidebar.header("关于本项目")
st.sidebar.info(
    """
    这是一个端到端的NLP项目，旨在通过AI技术赋能城市交通管理，自动分析海量社情民意，提升响应效率。
    
    **核心技术栈:**
    - **数据工程:** `requests`, `BeautifulSoup`, `pandas`
    - **智能标注:** LLM API (`通义千问`), `Prompt Engineering`
    - **模型训练:** `PyTorch`, `Hugging Face Transformers`
    - **基座模型:** `hfl/chinese-macbert-base`
    - **应用构建:** `Streamlit`, `Plotly`
    """
)
st.sidebar.markdown("---")
st.sidebar.write("**开发者:** 潘禹萌 秦赫")
st.sidebar.write("**GitHub:** https://github.com/mengzi667/NLP_TPnews.git")