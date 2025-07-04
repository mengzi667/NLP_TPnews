import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="城市交通舆情意图识别系统",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 自定义CSS样式 ---
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* 副标题样式 */
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* 卡片样式 */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* 意图类别标签样式 */
    .intent-label {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background: #f3f4f6;
        border-radius: 20px;
        font-size: 0.875rem;
        color: #374151;
        border: 1px solid #d1d5db;
    }
    
    /* 结果卡片样式 */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* 解读文本样式 */
    .interpretation-text {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        font-style: italic;
        color: #475569;
    }
    
    /* 隐藏Streamlit默认元素 */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 加载模型 ---
@st.cache_resource
def load_model():
    """从Hugging Face Hub加载训练好的模型和分词器。"""
    model_id = "Mengzi667/macbert-traffic-intent-classifier" 
    try:
        with st.spinner("🤖 正在从云端加载AI模型..."):
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        st.success("✅ 模型加载成功！", icon="🎉")
        return tokenizer, model
    except Exception as e:
        st.error("❌ 模型加载失败，请检查网络连接", icon="⚠️")
        st.error(f"错误详情: {e}")
        return None, None

tokenizer, model = load_model()

# --- 4. 页面标题和简介 ---
st.markdown('<h1 class="main-title">🚇 城市交通舆情意图识别系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">基于MacBERT深度学习模型，智能识别城市交通相关文本的核心意图<br>让城市交通管理更智能、更高效</p>', unsafe_allow_html=True)

# --- 5. 主应用界面 ---
if model is None:
    st.error("🚫 模型未能成功加载，应用无法正常工作。请刷新页面重试。", icon="⚠️")
else:
    # --- 预设的解读文案 ---
    INTERPRETATION_GUIDE = {
        "投诉": "模型识别出文本中含有强烈的负面情绪和对现状的不满，这通常指向一个需要被解决的问题。",
        "建议": "模型捕捉到了一个具体的改进想法或方案。这类反馈对优化服务非常有价值。",
        "咨询": "文本的核心是一个信息问询，用户希望获得一个客观的答案。",
        "求助": "这是一个明确的求助信号，通常与个人遇到的具体困难（如失物）相关。",
        "举报": "模型识别出了对某个具体违规行为或安全隐患的揭发，需要相关部门关注。",
        "正面反馈 (Positive Feedback)": "文本表达了明确的赞扬或感谢，是提升服务信心的重要来源。",
    }

    # 简化的单列布局
    st.markdown("### 📝 文本分析")
    
    # 自定义输入框样式
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        with st.form("intent_form", clear_on_submit=False):
            text_input = st.text_area(
                "请输入要分析的文本:",
                value="强烈建议地铁11号线在早高峰增加几班车，现在等一趟的时间也太长了！",
                height=120,
                placeholder="例如：上海地铁的空调是打算把人冻死吗？",
                help="💡 咨询 | 🆘 求助 | 😤 投诉 | 🚨 举报 | 💡 建议 | 👍 正面反馈"
            )
            submitted = st.form_submit_button("🚀 开始智能分析", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 6. 模型推理与结果展示 ---
    if submitted:
        if not text_input.strip():
            st.warning("⚠️ 请输入有效的文本内容后再进行分析。", icon="📝")
        else:
            with st.spinner('🤖 AI正在深度分析中，请稍候...'):
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1).flatten().tolist()
                labels = list(model.config.id2label.values())
                prob_df = pd.DataFrame({'意图类别': labels, '概率': probabilities}).sort_values(by='概率', ascending=False).reset_index(drop=True)

                st.markdown("---")
                st.markdown("### 📊 分析结果")
                
                # 创建结果展示区域
                result_col1, result_col2 = st.columns([1.5, 1], gap="large")
                
                with result_col1:
                    # 优化的条形图
                    fig = px.bar(
                        prob_df, 
                        x='概率', 
                        y='意图类别', 
                        orientation='h',
                        title='各意图类别概率分布',
                        text=[f'{p:.1%}' for p in prob_df['概率']],
                        color='概率',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending', 'showgrid': False, 'title': ""},
                        xaxis={'showgrid': True, 'gridcolor': 'rgba(0,0,0,0.1)', 'title': ""},
                        height=350,
                        showlegend=False,
                        title_x=0.5,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                with result_col2:
                    # 结果卡片
                    top_prediction = prob_df.iloc[0]
                    confidence = top_prediction['概率']
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h3 style="margin: 0 0 0.5rem 0;">🎯 识别结果</h3>
                        <h2 style="margin: 0 0 0.5rem 0;">{top_prediction['意图类别']}</h2>
                        <p style="margin: 0; font-size: 1.1rem;">置信度: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 智能解读
                    interpretation_text = INTERPRETATION_GUIDE.get(top_prediction['意图类别'], "这是一个通用反馈。")
                    st.markdown(f'<div class="interpretation-text">💡 <strong>智能解读:</strong><br>{interpretation_text}</div>', unsafe_allow_html=True)

                    # 详细数据展开
                    with st.expander("📊 详细数据", expanded=False):
                        for idx, row in prob_df.iterrows():
                            percentage = row['概率']
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin-bottom: 0.25rem; background: #f8fafc; border-radius: 6px;">
                                <span>{row['意图类别']}</span>
                                <span style="font-weight: bold; color: #667eea;">{percentage:.1%}</span>
                            </div>
                            """, unsafe_allow_html=True)

# --- 7. 简化的侧边栏 ---
with st.sidebar:
    st.markdown("### 🔬 项目信息")
    
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="margin-bottom: 0; color: #6b7280; font-size: 0.9rem;">
            基于MacBERT深度学习模型的城市交通舆情意图识别系统，
            自动分析社情民意，提升管理效率。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**🛠️ 核心技术**")
    st.markdown("""
    - **模型:** MacBERT + PyTorch
    - **优化:** 类别加权 + Optuna
    - **部署:** Streamlit + Plotly
    """)
    
    st.markdown("---")
    
    st.markdown("**👨‍💻 开发团队**")
    st.markdown("潘禹萌 & 秦赫")
    
    st.markdown("**🔗 项目链接**")
    st.markdown("[GitHub 仓库](https://github.com/mengzi667/NLP_TPnews)")
    
    st.markdown("---")
    st.info("💡 支持识别：咨询、求助、投诉、举报、建议、正面反馈")