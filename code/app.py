import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import os

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="城市交通舆情意图识别系统",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
    theme="light"
)

# --- 2. 加载模型 ---
@st.cache_resource
def load_model():
    """从Hugging Face Hub加载训练好的模型和分词器。"""
    model_id = "Mengzi667/macbert-traffic-intent-classifier" 
    try:
        st.info(f"正在从云端Hugging Face Hub加载模型: {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        st.success("模型和分词器已成功从云端加载！")
        return tokenizer, model
    except Exception as e:
        st.error(f"从Hugging Face Hub加载模型失败。请确认模型ID、仓库公开状态及网络连接。")
        st.error(f"错误详情: {e}")
        return None, None

tokenizer, model = load_model()

# --- 3. 页面标题和简介 ---
st.title("🚇 城市交通舆情意图识别系统")
st.markdown("""
欢迎使用本系统！这是一个基于 `MacBERT` 微调模型的自然语言处理应用，能够自动识别关于**城市交通**（地铁、公交、出租车等）的文本所表达的核心意图。
""")
st.markdown("---")

# --- 4. 主应用界面 ---
if model is None:
    st.warning("模型未能成功加载，应用无法正常工作。")
else:
    # --- 预设的解读文案 ---
    INTERPRETATION_GUIDE = {
        "投诉 (Complaint)": "模型识别出文本中含有强烈的负面情绪和对现状的不满，这通常指向一个需要被解决的问题。",
        "建议 (Suggestion)": "模型捕捉到了一个具体的改进想法或方案。这类反馈对优化服务非常有价值。",
        "咨询 (Consultation)": "文本的核心是一个信息问询，用户希望获得一个客观的答案。",
        "求助 (Request for Help)": "这是一个明确的求助信号，通常与个人遇到的具体困难（如失物）相关。",
        "举报 (Report)": "模型识别出了对某个具体违规行为或安全隐患的揭发，需要相关部门关注。",
        "正面反馈 (Positive Feedback)": "文本表达了明确的赞扬或感谢，是提升服务信心的重要来源。",
    }

    col1, col2 = st.columns([2, 1]) 

    with col1:
        st.header("✏️ 请输入要分析的文本:")
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

    # --- 5. 模型推理与结果展示 (全新布局) ---
    if submitted:
        if not text_input.strip():
            st.warning("请输入有效的文本内容后再进行分析。")
        else:
            with st.spinner('模型正在全力分析中，请稍候...'):
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1).flatten().tolist()
                labels = list(model.config.id2label.values())
                prob_df = pd.DataFrame({'意图类别': labels, '概率': probabilities}).sort_values(by='概率', ascending=False).reset_index(drop=True)

                st.markdown("---")
                st.header("📈 分析结果")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    # 左侧只保留最直观的条形图
                    fig = px.bar(
                        prob_df, x='概率', y='意图类别', orientation='h', 
                        title='各意图类别概率分布图', text_auto='.2%',
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        xaxis_title="概率",
                        yaxis_title="意图类别",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with res_col2:
                    # 右侧提供深度解读
                    top_prediction = prob_df.iloc[0]
                    st.metric(
                        label="**最高概率意图**", 
                        value=top_prediction['意图类别'],
                        help=f"模型对这个判断的置信度为 {top_prediction['概率']:.2%}"
                    )
                    
                    st.write("**结果解读:**")
                    interpretation_text = INTERPRETATION_GUIDE.get(top_prediction['意图类别'], "这是一个通用反馈。")
                    st.markdown(f"> {interpretation_text}")

                    with st.expander("查看详细概率数据"):
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
# --- 6. 侧边栏 (Sidebar) ---
st.sidebar.header("关于本项目")
st.sidebar.info(
    """
    这是一个端到端的NLP项目，旨在通过AI技术赋能城市交通管理，自动分析海量社情民意，提升响应效率。
    
    **核心技术栈:**
    - **智能标注:** LLM API (`通义千问`), Prompt Engineering
    - **模型训练:** `PyTorch`, `Hugging Face Transformers`
    - **模型选型:** `MacBERT` vs `RoBERTa` (A/B Test)
    - **性能优化:** 类别加权, 超参数搜索 (`Optuna`)
    - **应用构建:** `Streamlit`, `Plotly`
    """
)
st.sidebar.markdown("---")
st.sidebar.write("**开发者:** 潘禹萌 秦赫")
st.sidebar.write("**GitHub:** [https://github.com/mengzi667/NLP_TPnews](https://github.com/mengzi667/NLP_TPnews)")
