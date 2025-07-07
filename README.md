# 🚇 城市交通舆情意图识别系统 (Urban Traffic Intent Recognition System)

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff69b4.svg)

**一个基于大语言模型辅助标注和深度学习微调的、端到端的NLP项目，用于精准识别城市交通领域的公众反馈意图。**

---

## 🚀 在线体验 (Live Demo)

您可以访问部署在Streamlit Cloud上的在线应用，亲身体验模型的效果：

**[➡️ 点击这里，立即体验！](https://nlp-tpnews.streamlit.app/)**

![alt text](assets\image.png)

---

## 📖 项目简介 (Introduction)

在城市管理中，社交媒体上的海量公众反馈（如微博）是宝贵的“民意富矿”，但也带来了巨大的分析挑战。本项目旨在解决在缺乏大规模人工标注数据的情况下，如何快速、低成本地启动一个特定领域（城市交通）的文本分类任务。

项目的核心采用了一种**大语言模型（LLM）与传统模型微调相结合的混合策略**：
1.  首先，通过**提示工程 (Prompt Engineering)**，利用通义千问的零样本学习能力对原始语料进行自动化预标注，高效解决**数据冷启动**问题。
2.  然后，利用这份高质量的标注数据集，对一个轻量级的专用模型 (`hfl/chinese-macbert-base`) 进行高效微调和系统的性能优化。
3.  最终，将训练好的模型封装成一个可交互的Web应用，完成了从数据到产品的全流程闭环。

---

## ✨ 主要特色 (Key Features)

* **端到端实现：** 覆盖了从数据采集、智能标注、模型训练、超参数调优到最终应用部署的全链路。
* **创新标注策略：** 成功实践了**LLM辅助标注**的弱监督学习范式，极大地提升了数据处理效率。
* **深度模型优化：** 系统性地应用了**类别加权**来解决数据不平衡问题，并利用**Optuna**进行了自动化超参数搜索。
* **严谨的模型选型：** 对比了`MacBERT`和`RoBERTa`，并用数据选择了最优模型。
* **可交互展示：** 使用Streamlit和Plotly构建了美观、易用的Web应用，直观展示模型性能。

---

## 📊 最终模型性能 (Final Performance)

经过多轮迭代优化，最终的冠军模型（调优后的MacBERT）在独立的测试集上取得了非常出色的表现，证明了混合策略的有效性。

* **宏平均 F1-Score: 89.7%**
* **总体准确率 (Accuracy): 94.7%**

![最终混淆矩阵](assets\confusion_matrix.png)

---

## 🛠️ 技术栈 (Tech Stack)

* **数据工程:** `requests`, `BeautifulSoup`, `pandas`
* **智能标注:** LLM API (`通义千问`), `Prompt Engineering`
* **模型训练:** `PyTorch`, `Hugging Face Transformers`, `datasets`, `accelerate`
* **性能优化:** `scikit-learn` (类别权重), `Optuna` (超参数搜索)
* **应用构建:** `Streamlit`, `Plotly`
* **环境与版本控制:** `Conda`, `Git`

---

## ⚙️ 如何在本地运行 (Getting Started)

请按照以下步骤，在你的本地电脑上克隆并运行本项目。

**1. 克隆仓库:**
打开你的终端，运行以下命令将本项目克隆到本地。
```bash
git clone [https://github.com/mengzi667/NLP_TPnews.git](https://github.com/mengzi667/NLP_TPnews.git)
```

**2. 进入项目目录:**
```bash
cd NLP_TPnews
```

**3. 创建并激活Conda环境:**
本项目推荐使用Conda进行环境管理。以下命令会创建一个名为`nlp_app`的、使用Python 3.10的干净环境。
```bash
conda create -n nlp_app python=3.12
conda activate nlp_app
```

**4. 安装所有依赖:**
此命令会自动读取`requirements.txt`文件，并安装所有必需的Python库。
```bash
pip install -r requirements.txt
```

**5. 运行Web应用:**
一切准备就绪！运行以下命令来启动Streamlit交互式应用。
```bash
streamlit run app.py
```
应用启动后，你的浏览器会自动打开一个新的标签页，显示应用界面。

---

## 📂 项目结构 (Project Structure)

本项目采用标准的`src`目录结构，实现了源代码、应用和配置的分离。
```
.
├── 📜 README.md
├── 📝 requirements.txt
├── .gitignore
├── 🚀 app.py
├── check_envs.py
└── 📂 src/
    ├── 📂 data_collection/
    ├── 📂 data_processing/
    ├── 📂 training/
    └── 📂 evaluation/
```

---

## 🙏 致谢 (Acknowledgements)

* 感谢**Hugging Face**社区提供了强大的开源模型和工具库。
* 感谢**阿里云灵积平台**提供了通义千问的API服务。
* 感谢**Streamlit**团队让数据应用的构建变得如此简单。
* 感谢秦子在数据获取、清洗和标注过程中的贡献