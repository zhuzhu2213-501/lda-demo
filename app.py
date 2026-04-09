#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
LDA 文本主题分析教学演示 - 主程序
=============================================================================

项目名称：LDA 文本主题分析：从原理到实践的史上最详细交互式教学演示
版本：v1.0
作者：AI Assistant

功能说明：
    这是一个专为教学设计的 LDA（Latent Dirichlet Allocation，潜在狄利克雷分配）
    主题模型交互式学习工具。通过丰富的可视化、逐步演示和交互式实验，
    帮助学习者从原理到实践全面掌握 LDA 文本主题分析技术。

模块列表（共13个核心教学模块）：
    1. 欢迎页/导航页
    2. 什么是主题模型
    3. LDA的直觉理解
    4. 数学与概率图模型
    5. 文本预处理教学
    6. 从零运行LDA案例
    7. 参数实验室
    8. 可视化中心
    9. 结果解释教学
    10. 模型评估与选参
    11. LDA局限与替代方法
    12. 应用案例区
    13. 练习与自测区

使用方法：
    1. 安装依赖：pip install -r requirements.txt
    2. 运行程序：streamlit run app.py
    3. 浏览器访问 http://localhost:8501

依赖说明：
    - streamlit: Web框架
    - pandas, numpy: 数据处理
    - matplotlib, plotly: 可视化
    - sklearn: LDA实现
    - jieba: 中文分词
    - wordcloud: 词云
    - pyLDAvis: 交互式LDA可视化

=============================================================================
"""

# ============================================================================
# 导入必要的库
# ============================================================================

import streamlit as st          # Web应用框架
import pandas as pd             # 数据处理
import numpy as np              # 数值计算
import re                       # 正则表达式
import os                       # 文件路径操作
import sys                      # 系统操作
import base64                  # 文件编码
from io import StringIO         # 字符串IO

# 可视化库
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免Streamlit环境报错

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP和主题建模库
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# 中文分词
import jieba

# 词云
from wordcloud import WordCloud

# 进度显示
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置设置
# ============================================================================

# 设置页面配置 - 定义Web应用的基本属性
st.set_page_config(
    page_title="LDA 主题分析教学演示",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置中文字体支持 - 确保matplotlib可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义项目根目录（相对于当前文件的位置）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
STOPWORDS_DIR = os.path.join(CURRENT_DIR, 'stopwords')

# ============================================================================
# 辅助函数
# ============================================================================

def load_stopwords(filename, is_chinese=False):
    """
    加载停用词表
    
    参数:
        filename: 停用词文件名
        is_chinese: 是否为中文停用词
    
    返回:
        set: 停用词集合
    """
    filepath = os.path.join(STOPWORDS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return set(words)
    except FileNotFoundError:
        return set()

def load_sample_data(dataset_name):
    """
    加载示例数据集
    
    参数:
        dataset_name: 数据集名称 ('news', 'reviews', 'academic', 'chinese')
    
    返回:
        DataFrame或list: 加载的数据
    """
    if dataset_name == 'chinese':
        filepath = os.path.join(DATA_DIR, 'chinese_sample.txt')
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            documents = []
            for line in lines:
                parts = line.split('|')
                if len(parts) >= 3:
                    documents.append({
                        'id': parts[0],
                        'category': parts[1],
                        'text': '|'.join(parts[2:])
                    })
            return documents
        except FileNotFoundError:
            return []
    else:
        csv_files = {
            'news': 'news_sample.csv',
            'reviews': 'reviews_sample.csv',
            'academic': 'academic_sample.csv'
        }
        filepath = os.path.join(DATA_DIR, csv_files.get(dataset_name, ''))
        try:
            # 移除注释行
            df = pd.read_csv(filepath, comment='#')
            return df
        except FileNotFoundError:
            return pd.DataFrame()

def preprocess_english_text(text, stop_words):
    """
    英文文本预处理：分词、去除停用词、只保留字母单词
    
    参数:
        text: 原始文本
        stop_words: 停用词集合
    
    返回:
        str: 处理后的文本
    """
    # 转小写
    text = text.lower()
    # 只保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 分词
    words = text.split()
    # 去除停用词和短词
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def preprocess_chinese_text(text, stop_words):
    """
    中文文本预处理：分词、去除停用词
    
    参数:
        text: 原始文本
        stop_words: 停用词集合
    
    返回:
        str: 处理后的文本
    """
    # jieba分词
    words = jieba.cut(text)
    # 去除停用词和单字
    words = [w.strip() for w in words if w.strip() and w not in stop_words and len(w) > 1]
    return ' '.join(words)

def create_lda_model(documents, n_topics=5, alpha='auto', beta='auto', max_iter=20):
    """
    创建并训练LDA模型
    
    参数:
        documents: 文档列表（已预处理的文本）
        n_topics: 主题数量K
        alpha: 文档-主题分布的先验参数
        beta: 主题-词分布的先验参数
        max_iter: 最大迭代次数
    
    返回:
        tuple: (训练好的模型, 词向量器, 文档-主题矩阵, 主题-词矩阵)
    """
    # 创建词袋模型
    vectorizer = CountVectorizer(
        max_df=0.95,      # 忽略出现在95%以上文档中的词
        min_df=2,          # 至少出现在2个文档中
        max_features=1000  # 最多保留1000个词
    )
    
    # 转换文档为词频矩阵
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    # 创建LDA模型
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        doc_topic_prior=alpha if alpha != 'auto' else None,
        topic_word_prior=beta if beta != 'auto' else None,
        max_iter=max_iter,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    lda_model.fit(doc_term_matrix)
    
    # 获取文档-主题分布
    doc_topic_dist = lda_model.transform(doc_term_matrix)
    
    # 获取主题-词分布
    topic_word_dist = lda_model.components_
    
    return lda_model, vectorizer, doc_topic_dist, topic_word_dist

def get_top_words_per_topic(model, feature_names, n_top_words=10):
    """
    获取每个主题的Top N关键词
    
    参数:
        model: 训练好的LDA模型
        feature_names: 特征名称（词汇表）
        n_top_words: 每个主题返回的词数
    
    返回:
        dict: {主题索引: [(词, 权重), ...]}
    """
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-n_top_words-1:-1]
        top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
        topics[topic_idx] = top_words
    return topics

def calculate_perplexity(model, doc_term_matrix):
    """
    计算模型困惑度
    
    参数:
        model: 训练好的LDA模型
        doc_term_matrix: 文档-词矩阵
    
    返回:
        float: 困惑度分数
    """
    return model.perplexity(doc_term_matrix)

def calculate_coherence(topics, doc_term_matrix, vectorizer, coherence_type='c_v'):
    """
    计算主题一致性分数（简化版）
    
    注意：完整的 coherence 需要 gensim 库，这里使用简化版实现
    
    参数:
        topics: 主题词列表
        doc_term_matrix: 文档-词矩阵
        vectorizer: 词向量器
        coherence_type: 一致性类型
    
    返回:
        float: 一致性分数
    """
    # 简化的PMI-based coherence
    # 在实际应用中建议使用 gensim 的 CoherenceModel
    
    feature_names = vectorizer.get_feature_names_out()
    doc_term_array = doc_term_matrix.toarray()
    
    coherence_scores = []
    for topic_words in topics.values():
        word_indices = [np.where(feature_names == word)[0][0] for word, _ in topic_words if word in feature_names]
        
        if len(word_indices) < 2:
            continue
            
        # 计算共现分数
        co_occurrence = 0
        total_pairs = 0
        for i, idx1 in enumerate(word_indices):
            for idx2 in word_indices[i+1:]:
                # 计算两个词在同一个文档中出现的次数
                co_doc = np.sum((doc_term_array[:, idx1] > 0) & (doc_term_array[:, idx2] > 0))
                if co_doc > 0:
                    co_occurrence += np.log(co_doc + 1)
                total_pairs += 1
        
        if total_pairs > 0:
            coherence_scores.append(co_occurrence / total_pairs)
    
    return np.mean(coherence_scores) if coherence_scores else 0

def create_pyldavis_data(model, vectorizer, doc_term_matrix):
    """
    准备pyLDAvis可视化数据
    
    参数:
        model: LDA模型
        vectorizer: 词向量器
        doc_term_matrix: 文档-词矩阵
    
    返回:
        pyLDAvis.PreparedData: 可视化数据
    """
    try:
        import pyLDAvis
        import pyLDAvis./sklearn
        
        # 准备数据
        prepared = pyLDAvis./sklearn.prepare(
            model, 
            doc_term_matrix, 
            vectorizer,
            mds='tsne'
        )
        return prepared
    except Exception as e:
        st.error(f"pyLDAvis准备失败: {str(e)}")
        return None

def save_pyldavis_html(prepared, filename='lda_vis.html'):
    """
    保存pyLDAvis为HTML文件
    
    参数:
        prepared: pyLDAvis数据
        filename: 保存的文件名
    
    返回:
        str: HTML内容
    """
    try:
        import pyLDAvis
        html = pyLDAvis.prepared_data_to_html(prepared)
        return html
    except Exception as e:
        return None

def create_wordcloud(topic_word_dist, feature_names, topic_idx=0):
    """
    为单个主题创建词云
    
    参数:
        topic_word_dist: 主题-词分布矩阵
        feature_names: 词汇表
        topic_idx: 主题索引
    
    返回:
        WordCloud: 词云对象
    """
    # 获取该主题的词权重
    word_weights = {feature_names[i]: topic_word_dist[topic_idx][i] 
                   for i in range(len(feature_names))}
    
    # 创建词云
    wc = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=50,
        colormap='viridis',
        random_state=42
    ).generate_from_frequencies(word_weights)
    
    return wc

def create_download_link(object_to_download, download_filename, link_text):
    """
    创建文件下载链接
    
    参数:
        object_to_download: 要下载的对象（HTML字符串或文件路径）
        download_filename: 下载文件名
        link_text: 链接文本
    
    返回:
        str: HTML下载链接
    """
    if isinstance(object_to_download, str):
        b64 = base64.b64encode(object_to_download.encode()).decode()
    else:
        b64 = base64.b64encode(object_to_download).decode()
    
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{link_text}</a>'

# ============================================================================
# 页面组件函数
# ============================================================================

def render_header():
    """渲染页面标题和简介"""
    st.title("📚 LDA 文本主题分析")
    st.markdown("### 从原理到实践的史上最详细交互式教学演示")
    st.markdown("---")

def render_sidebar():
    """
    渲染侧边栏导航
    
    返回:
        str: 选择的模块名称
    """
    st.sidebar.title("📑 学习导航")
    st.sidebar.markdown("---")
    
    # 简介信息
    st.sidebar.info("""
    **LDA (Latent Dirichlet Allocation)**
    
    潜在狄利克雷分配，一种经典的主题模型算法，用于发现文档集合中的潜在主题结构。
    """)
    
    st.sidebar.markdown("---")
    
    # 模块选择
    modules = {
        "🏠 欢迎页": "welcome",
        "📖 什么是主题模型": "what_is_topic",
        "🧠 LDA直觉理解": "lda_intuition",
        "🔢 数学与概率图模型": "math_model",
        "🔧 文本预处理教学": "preprocessing",
        "🚀 从零运行LDA案例": "lda_tutorial",
        "⚗️ 参数实验室": "parameter_lab",
        "📊 可视化中心": "visualization",
        "💡 结果解释教学": "result_interpretation",
        "📈 模型评估与选参": "model_evaluation",
        "⚠️ LDA局限与替代": "alternatives",
        "💼 应用案例区": "applications",
        "📝 练习与自测": "quiz"
    }
    
    selected = st.sidebar.radio(
        "选择学习模块",
        list(modules.keys()),
        index=0,
        format_func=lambda x: x
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**学习进度建议**")
    st.sidebar.markdown("""
    1. 先看欢迎页了解整体框架
    2. 按顺序学习模块2-5打牢基础
    3. 模块6-7动手实践
    4. 模块8-10深入理解
    5. 模块11-13自由探索
    """)
    
    return modules[selected]

# ============================================================================
# 模块1: 欢迎页
# ============================================================================

def module_welcome():
    """欢迎页模块"""
    st.header("🏠 欢迎学习 LDA 主题模型")
    
    # 项目介绍
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 关于本项目")
        st.markdown("""
        **欢迎来到 LDA 主题分析教学演示！**
        
        这是一个专为教学设计的交互式学习工具，旨在帮助您：
        - 🔍 **理解** 主题模型的核心概念
        - 🎯 **掌握** LDA 算法的原理
        - 💻 **实践** 文本主题分析技能
        - 📊 **应用** 到您的实际项目中
        """)
        
        st.subheader("🎯 学习目标")
        st.markdown("""
        完成本课程后，您将能够：
        
        1. **理解主题模型的基本概念** - 知道什么是主题、为什么需要主题模型
        2. **掌握 LDA 的数学原理** - 理解 Dirichlet 分布、生成过程
        3. **熟练进行文本预处理** - 分词、去停用词、构建词袋模型
        4. **训练和评估 LDA 模型** - 调参、可视化、结果解读
        5. **应用于实际场景** - 新闻分类、评论分析、学术研究等
        """)
    
    with col2:
        st.markdown("### 📈 课程统计")
        st.metric("核心模块", "13 个")
        st.metric("教学功能", "30+ 个")
        st.metric("可视化图表", "20+ 种")
        st.metric("内置数据集", "4 个")
    
    st.markdown("---")
    
    # 学习路径
    st.subheader("🛤️ 学习路径")
    
    path_col1, path_col2, path_col3 = st.columns(3)
    
    with path_col1:
        st.markdown("""
        ### 阶段一：概念入门
        ⏱️ 建议时间：30分钟
        
        | 模块 | 内容 |
        |------|------|
        | 什么是主题模型 | 目标、概念、与其他方法对比 |
        | LDA直觉理解 | 调色盘/菜谱比喻 |
        | 数学基础 | 概率图模型、Dirichlet分布 |
        """)
    
    with path_col2:
        st.markdown("""
        ### 阶段二：动手实践
        ⏱️ 建议时间：60分钟
        
        | 模块 | 内容 |
        |------|------|
        | 文本预处理 | 分词、去停用词、词袋模型 |
        | 从零运行LDA | 完整案例演示 |
        | 参数实验室 | K值、alpha、beta调节 |
        """)
    
    with path_col3:
        st.markdown("""
        ### 阶段三：深入应用
        ⏱️ 建议时间：60分钟
        
        | 模块 | 内容 |
        |------|------|
        | 可视化中心 | 多种可视化方法 |
        | 模型评估 | 困惑度、一致性指标 |
        | 实战案例 | 多领域应用 |
        """)
    
    st.markdown("---")
    
    # 快速开始
    st.subheader("🚀 快速开始")
    
    quick_start_col1, quick_start_col2 = st.columns([1, 1])
    
    with quick_start_col1:
        st.info("""
        **💡 建议的学习方式：**
        
        1. 从左侧导航栏选择模块
        2. 先阅读概念解释，理解原理
        3. 观看动画演示，建立直观认识
        4. 动手实验参数，观察效果
        5. 完成练习题，巩固所学
        """)
    
    with quick_start_col2:
        st.success("""
        **🎁 本项目特点：**
        
        - ✅ 详细的代码注释
        - ✅ 交互式参数调节
        - ✅ 多种可视化展示
        - ✅ 真实数据集练习
        - ✅ 即时反馈与验证
        """)
    
    st.markdown("---")
    
    # 常见问题
    with st.expander("❓ 常见问题 FAQ"):
        st.markdown("""
        **Q: 需要什么样的基础才能学习本课程？**
        
        A: 建议具备基本的 Python 编程能力，了解一点概率统计知识会更好，但不是必需的。
        
        **Q: 学完后能达到什么水平？**
        
        A: 您将能够独立使用 LDA 进行文本主题分析，理解结果的含义，并应用到实际项目中。
        
        **Q: 可以使用自己的数据吗？**
        
        A: 当然可以！本项目支持上传自定义文本数据进行主题分析。
        
        **Q: pyLDAvis 无法显示怎么办？**
        
        A: pyLDAvis 需要浏览器支持 WebGL，请确保使用现代浏览器（如 Chrome、Firefox、Edge）。
        """)

# ============================================================================
# 模块2: 什么是主题模型
# ============================================================================

def module_what_is_topic():
    """什么是主题模型模块"""
    st.header("📖 什么是主题模型")
    
    # 概念引入
    st.subheader("🎯 主题模型的定义")
    
    definition_col1, definition_col2 = st.columns([1, 1])
    
    with definition_col1:
        st.markdown("""
        ### 📚 学术定义
        
        **主题模型 (Topic Model)** 是一类 **无监督机器学习** 算法，
        用于发现文档集合中隐藏的 **语义结构（主题）**。
        
        **核心思想**：每篇文档是多个主题的 **概率混合**，
        每个主题是词汇表中词的 **概率分布**。
        """)
    
    with definition_col2:
        st.markdown("""
        ### 🗣️ 通俗解释
        
        想象你有一堆杂志文章，主题模型就像是：
        
        > 📰 一位 **智能图书管理员**，能够自动识别：
        > - 每篇文章在讲什么主题（如：科技、体育、娱乐）
        > - 每个主题由哪些关键词代表
        > - 每篇文章属于各个主题的可能性
        """)
    
    st.markdown("---")
    
    # 问题背景
    st.subheader("🤔 为什么要用主题模型？")
    
    problem_col1, problem_col2 = st.columns([1, 1])
    
    with problem_col1:
        st.markdown("""
        ### 传统方法的局限
        
        | 方法 | 局限 |
        |------|------|
        | 关键词匹配 | 无法理解语义，容易误判 |
        | TF-IDF | 只能找到"重要词"，不能发现主题 |
        | 人工标注 | 耗时耗力，主观性强 |
        | 分类模型 | 需要预先定义类别，无法发现新主题 |
        """)
    
    with problem_col2:
        st.markdown("""
        ### 主题模型的优势
        
        | 优势 | 说明 |
        |------|------|
        | 自动发现 | 无需人工标注，自动识别主题 |
        | 概率解释 | 每个词、文档都有概率值 |
        | 软聚类 | 一篇文档可属于多个主题 |
        | 语义理解 | 发现词汇间的语义关联 |
        | 可解释性 | 主题由关键词定义，可理解 |
        """)
    
    st.markdown("---")
    
    # 主题模型 vs 其他方法
    st.subheader("⚖️ 主题模型 vs 其他方法对比")
    
    # 创建对比表格
    comparison_data = {
        '方法': ['关键词匹配', 'TF-IDF', '聚类分析', '主题模型', '深度学习'],
        '是否需要标注': ['否', '否', '可选', '否', '可选', ],
        '发现新主题': ['❌', '❌', '✅', '✅', '✅'],
        '多主题支持': ['❌', '❌', '✅', '✅', '✅'],
        '概率解释': ['❌', '⚠️', '⚠️', '✅', '⚠️'],
        '语义理解': ['❌', '❌', '⚠️', '⚠️', '✅'],
        '计算效率': ['快', '快', '中等', '中等', '慢'],
        '可解释性': ['高', '高', '中等', '高', '低']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("""
    **图例**: ✅ 支持 | ❌ 不支持 | ⚠️ 部分支持
    """)
    
    st.markdown("---")
    
    # 主题模型的应用场景
    st.subheader("📱 主题模型的应用场景")
    
    app_col1, app_col2, app_col3 = st.columns(3)
    
    with app_col1:
        st.markdown("""
        ### 📰 信息检索
        - 新闻分类与推荐
        - 文档聚类
        - 相似文档发现
        - 搜索引擎优化
        """)
    
    with app_col2:
        st.markdown("""
        ### 💼 商业分析
        - 客户评论分析
        - 社交媒体监控
        - 市场研究
        - 品牌声誉管理
        """)
    
    with app_col3:
        st.markdown("""
        ### 🔬 学术研究
        - 文献综述
        - 研究趋势发现
        - 知识图谱构建
        - 论文查重与相似度
        """)
    
    st.markdown("---")
    
    # 常见的主题模型算法
    with st.expander("📚 常见主题模型算法一览"):
        st.markdown("""
        ### 主要主题模型算法
        
        | 算法 | 全称 | 特点 |
        |------|------|------|
        | **LDA** | Latent Dirichlet Allocation | 最经典、应用最广 |
        | **pLSA** | Probabilistic Latent Semantic Analysis | LDA的前身 |
        | **CTM** | Correlated Topic Model | 支持主题间相关性 |
        | **LFTM** | Latent Feature Topic Model | 结合主题模型和矩阵分解 |
        | **NMF** | Non-negative Matrix Factorization | 另一种矩阵分解方法 |
        | **BERTopic** | BERT + Topic Modeling | 结合深度学习的现代方法 |
        
        本教程主要聚焦于 **LDA**，因为它：
        1. 原理清晰，易于理解
        2. 发展成熟，工具完善
        3. 效果好，稳定可靠
        4. 应用广泛，案例丰富
        """)

# ============================================================================
# 模块3: LDA直觉理解
# ============================================================================

def module_lda_intuition():
    """LDA直觉理解模块"""
    st.header("🧠 LDA 的直觉理解")
    
    st.markdown("""
    LDA（潜在狄利克雷分配）最初听起来可能很复杂，
    但通过一些生动的比喻，我们可以建立直观的理解。
    """)
    
    # 比喻选择
    metaphor = st.selectbox(
        "选择比喻类型",
        ["🎨 调色盘比喻", "👨‍🍳 菜谱比喻", "🎰 抽签盒比喻"]
    )
    
    if metaphor == "🎨 调色盘比喻":
        st.subheader("🎨 比喻一：调色盘")
        
        metaphor_col1, metaphor_col2 = st.columns([1, 1])
        
        with metaphor_col1:
            st.markdown("""
            ### 核心思想
            
            把 LDA 想象成一位 **画家** 创作一幅画的过程：
            
            1. **画布（文档）** = 画家在画布上作画
            2. **调色盘（主题混合）** = 画家从调色盘选择颜料
            3. **颜料（词汇）** = 不同颜色混合产生最终画作
            
            **画家如何创作？**
            - 首先，决定用哪些颜色（选择主题）
            - 然后，决定每种颜色的用量（主题权重）
            - 最后，把颜料涂在画布上（生成词汇）
            """)
        
        with metaphor_col2:
            st.markdown("""
            ### 🎬 动画演示
            
            假设有一幅画包含3种"主题颜色"：
            
            | 主题 | 代表色 | 关键词示例 |
            |------|--------|-----------|
            | 科技 | 🔵 蓝色系 | 电脑、代码、数据 |
            | 体育 | 🟢 绿色系 | 比赛、球员、进球 |
            | 娱乐 | 🟣 紫色系 | 电影、音乐、明星 |
            
            当我们看这幅画时，可以看到混合后的颜色，
            但 LDA 能帮我们**反推**出用了哪些原始颜色！
            """)
        
        # 交互式演示
        st.markdown("### 🖱️ 交互式演示：颜色混合")
        
        mix_col1, mix_col2, mix_col3 = st.columns(3)
        
        with mix_col1:
            tech_ratio = st.slider("科技主题比例", 0.0, 1.0, 0.5, 0.1)
        with mix_col2:
            sports_ratio = st.slider("体育主题比例", 0.0, 1.0, 0.3, 0.1)
        with mix_col3:
            ent_ratio = st.slider("娱乐主题比例", 0.0, 1.0, 0.2, 0.1)
        
        # 归一化
        total = tech_ratio + sports_ratio + ent_ratio
        if total > 0:
            tech_norm = tech_ratio / total
            sports_norm = sports_ratio / total
            ent_norm = ent_ratio / total
            
            st.markdown(f"""
            **归一化后的主题分布：**
            - 科技: {tech_norm:.1%}
            - 体育: {sports_norm:.1%}
            - 娱乐: {ent_norm:.1%}
            """)
            
            # 显示混合效果
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=['科技', '体育', '娱乐'],
                values=[tech_norm, sports_norm, ent_norm],
                marker_colors=['#3498db', '#2ecc71', '#9b59b6'],
                hole=0.4
            ))
            fig.update_layout(title="文档的主题混合比例")
            st.plotly_chart(fig, use_container_width=True)
    
    elif metaphor == "👨‍🍳 菜谱比喻":
        st.subheader("👨‍🍳 比喻二：菜谱制作")
        
        metaphor_col1, metaphor_col2 = st.columns([1, 1])
        
        with metaphor_col1:
            st.markdown("""
            ### 核心思想
            
            把 LDA 想象成 **餐厅厨师** 根据"神秘菜谱"做菜的过程：
            
            1. **菜谱本（主题）** = 餐厅有若干固定菜谱
            2. **点菜（文档-主题分布）** = 每桌客人按不同比例点菜
            3. **食材（词汇）** = 每道菜由特定食材组成
            
            **做菜过程（生成过程）：**
            1. 客人看一眼菜谱本，决定点什么菜（选主题）
            2. 客人按自己口味决定各菜的比例（主题权重）
            3. 厨师根据菜谱准备食材（生成词汇）
            """)
        
        with metaphor_col2:
            st.markdown("""
            ### 🍳 具体例子
            
            **餐厅有3道"主题菜"：**
            
            | 主题菜 | 主要食材（关键词） |
            |--------|---------------------|
            | 粤菜 | 鲜虾、清蒸、鱼 |
            | 川菜 | 辣椒、花椒、麻辣 |
            | 西餐 | 牛排、沙拉、红酒 |
            
            **客人A的点单：**
            - 70% 粤菜（清淡）
            - 20% 川菜（微辣）
            - 10% 西餐（偶尔换口味）
            
            客人A的"文档"就会包含很多"鲜虾"、"清蒸"这样的词！
            """)
        
        # 交互式演示
        st.markdown("### 🖱️ 交互式演示：模拟点菜")
        
        order_col1, order_col2, order_col3 = st.columns(3)
        
        with order_col1:
            canton_ratio = st.slider("粤菜比例", 0.0, 1.0, 0.4, 0.1, key="canton")
        with order_col2:
            sichuan_ratio = st.slider("川菜比例", 0.0, 1.0, 0.4, 0.1, key="sichuan")
        with order_col3:
            western_ratio = st.slider("西餐比例", 0.0, 1.0, 0.2, 0.1, key="western")
        
        total = canton_ratio + sichuan_ratio + western_ratio
        if total > 0:
            st.markdown(f"""
            **点单结果：**
            - 粤菜: {canton_ratio/total:.1%}
            - 川菜: {sichuan_ratio/total:.1%}
            - 西餐: {western_ratio/total:.1%}
            
            **预计会出现的"食材词汇"：**
            - 主要: 鲜虾、清蒸、鱼、生抽
            - 次要: 辣椒、花椒、豆瓣酱
            - 偶尔: 牛排、沙拉、奶酪
            """)
    
    else:  # 抽签盒比喻
        st.subheader("🎰 比喻三：抽签盒")
        
        metaphor_col1, metaphor_col2 = st.columns([1, 1])
        
        with metaphor_col1:
            st.markdown("""
            ### 核心思想
            
            把 LDA 想象成一个 **神奇的抽签盒**：
            
            1. **抽签盒（文档）** = 里面有很多签
            2. **签上的词（词汇）** = 抽到不同的词
            3. **颜色分类（主题）** = 不同颜色的签代表不同主题
            
            **工作原理：**
            - 盒子里有 N 堆签（主题），每堆颜色不同
            - 每堆签上有不同的词，但比例不同
            - 每次抽签，先选一堆（主题），再抽一根（词）
            - 抽完放回，重复多次
            """)
        
        with metaphor_col2:
            st.markdown("""
            ### 🎲 具体演示
            
            **假设有2个主题的签盒：**
            
            | 主题 | 签的颜色 | 签上的词（按比例） |
            |------|----------|-------------------|
            | 主题A | 🔴 红色 | 电脑(30%), 代码(25%), 软件(20%)... |
            | 主题B | 🔵 蓝色 | 比赛(28%), 球员(25%), 进球(22%)... |
            
            **抽签过程：**
            1. 随机选一个主题（按文档的主题分布）
            2. 从该主题的签堆中抽一根
            3. 记录抽到的词
            4. 放回，重复
            
            最终抽到的词就构成了"文档"！
            """)
        
        # 交互式演示
        st.markdown("### 🖱️ 交互式演示：模拟抽签")
        
       抽签次数 = st.slider("抽签次数", 10, 100, 50, 5)
        
        if st.button("开始抽签！"):
            # 模拟抽签
            np.random.seed(42)
            topic_a_prob = 0.6
            topic_a_words = ["电脑", "代码", "软件", "数据", "系统", "程序", "网络", "算法"]
            topic_b_words = ["比赛", "球员", "进球", "球队", "教练", "体育", "运动", "冠军"]
            
            results = []
            for _ in range(抽签次数):
                if np.random.random() < topic_a_prob:
                    results.append(np.random.choice(topic_a_words))
                else:
                    results.append(np.random.choice(topic_b_words))
            
            word_counts = Counter(results)
            
            st.markdown("**抽签结果统计：**")
            result_df = pd.DataFrame([
                {"词汇": word, "出现次数": count} 
                for word, count in word_counts.most_common()
            ])
            st.dataframe(result_df, use_container_width=True)
            
            st.markdown(f"""
            **观察结果：**
            - "电脑"、"代码"等词出现较多 → 说明这篇"文档"可能偏向主题A
            - 这就是 LDA 的**逆过程**：从观察到的词推断背后的主题结构
            """)
    
    st.markdown("---")
    
    # LDA的关键特点
    st.subheader("✨ LDA 的关键特点")
    
    feature_col1, feature_col2 = st.columns([1, 1])
    
    with feature_col1:
        st.markdown("""
        ### 📌 软聚类 (Soft Clustering)
        
        传统聚类：一篇文章要么属于A类，要么属于B类
        
        **LDA聚类**：一篇文章可以同时属于多个主题！
        - 这篇文章 60% 是科技，30% 是商业，10% 是娱乐
        - 更符合实际，文章很少只谈论单一主题
        """)
    
    with feature_col2:
        st.markdown("""
        ### 📌 词的概率分布
        
        传统观点：词要么"属于"某个主题，要么"不属于"
        
        **LDA观点**：每个词都以一定概率属于多个主题
        - "Java" 80% 属于编程主题，15% 属于咖啡主题
        - "Apple" 70% 属于科技主题，20% 属于水果主题
        - 上下文决定词的具体含义
        """)
    
    # 常见误解
    st.markdown("---")
    with st.expander("⚠️ 常见误解"):
        st.markdown("""
        ### LDA 使用中的常见误区
        
        | 误解 | 真相 |
        |------|------|
        | 主题是离散的标签 | 主题是词汇的概率分布，边界是模糊的 |
        | K越大越好 | K太大会导致过拟合，主题难以解释 |
        | 主题是互不重叠的 | 主题之间可以有词汇重叠 |
        | 每次运行结果都相同 | 使用随机种子控制可复现性 |
        | LDA能理解语义 | LDA只基于词的共现，不真正"理解"语义 |
        """)

# ============================================================================
# 模块4: 数学与概率图模型
# ============================================================================

def module_math_model():
    """数学与概率图模型模块"""
    st.header("🔢 数学与概率图模型")
    
    st.markdown("""
    本模块将深入 LDA 的数学基础。如果您觉得太难，可以先跳过，
    等有了更多实践后再回头学习。
    """)
    
    # 变量定义
    st.subheader("📝 变量定义")
    
    st.markdown("""
    在 LDA 中，我们定义以下变量：
    
    | 符号 | 含义 | 示例 |
    |------|------|------|
    | $D$ | 文档数量 | 1000篇文档 |
    | $K$ | 主题数量 | 5个主题 |
    | $V$ | 词汇表大小 | 5000个词 |
    | $\\alpha$ | 文档-主题分布的先验参数 | 0.1 |
    | $\\beta$ | 主题-词分布的先验参数 | 0.01 |
    | $\\theta_d$ | 文档d的主题分布 | [0.2, 0.5, 0.3] |
    | $\\phi_k$ | 主题k的词分布 | {"电脑":0.1, "代码":0.08, ...} |
    | $z_{d,n}$ | 文档d第n个词的主题分配 | topic 2 |
    | $w_{d,n}$ | 文档d第n个词 | "电脑" |
    """)
    
    st.markdown("---")
    
    # 生成过程
    st.subheader("🔄 LDA 生成过程")
    
    process_col1, process_col2 = st.columns([1, 1])
    
    with process_col1:
        st.markdown("""
        ### 📜 数学表述
        
        对于每篇文档 $d$：
        
        1. **从 Dirichlet 分布采样文档-主题分布**：
           $\\theta_d \\sim \\text{Dir}(\\alpha)$
        
        2. **对于文档中的每个词 $w_n$**：
           - 从多项分布采样一个主题：$z_n \\sim \\text{Multi}(\\theta_d)$
           - 从该主题的词分布采样词：$w_n \\sim \\text{Multi}(\\phi_{z_n})$
        
        **核心思想**：
        > 写文章时，我们先确定要写哪些主题（按一定比例），
        > 然后每个主题"召唤"出相关的词汇。
        """)
    
    with process_col2:
        st.markdown("""
        ### 🖥️ 算法步骤
        
        ```
        对于每篇文档 d:
            1. 采样文档的主题分布 θ_d
               θ_d ~ Dirichlet(α)
            
            2. 对于文档中的每个词 w:
               a. 采样一个主题 z
                  z ~ Multi(θ_d)
               
               b. 采样一个词 w
                  w ~ Multi(φ_z)
        
        输出: θ (文档-主题) 和 φ (主题-词)
        ```
        
        **这就是 LDA 的"正向生成过程"**
        """)
    
    # 可视化生成过程
    st.markdown("### 🎬 生成过程可视化")
    
    # 简单动画演示
    gen_step = st.radio(
        "选择生成步骤",
        ["步骤1: Dirichlet分布", "步骤2: 采样主题", "步骤3: 生成词汇", "完整过程"]
    )
    
    if gen_step == "步骤1: Dirichlet分布":
        st.markdown("""
        **步骤1：采样文档的主题分布**
        
        Dirichlet 分布是多项分布的共轭先验，它的输出是一个 **概率向量**。
        
        想象一个骰子：
        - 普通骰子：每个面概率相同 (1/6)
        - Dirichlet：生成"加权骰子"，每个面概率不同
        
        $\\alpha$ 参数控制分布的"稀疏性"：
        - $\\alpha$ 大 → 各主题概率接近（文档主题分布均匀）
        - $\\alpha$ 小 → 少数主题主导（文档主题分布集中）
        """)
        
        # 绘制 Dirichlet 分布可视化
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=['α=10 (均匀)', 'α=1 (中等)', 'α=0.1 (集中)'],
                           specs=[[{'type': 'ternary'}, {'type': 'ternary'}, {'type': 'ternary'}]])
        
        for i, alpha in enumerate([10, 1, 0.1]):
            np.random.seed(42)
            samples = np.random.dirichlet([alpha, alpha, alpha], 1)[0]
            
            fig.add_trace(go.Scatterternary({
                'a': [samples[0]],
                'b': [samples[1]],
                'c': [samples[2]],
                'mode': 'markers',
                'marker': {'size': 15, 'color': 'blue'}
            }), row=1, col=i+1)
        
        fig.update_ternaries(
            {'aaxis': '主题A', 'baxis': '主题B', 'caxis': '主题C'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif gen_step == "步骤2: 采样主题":
        st.markdown("""
        **步骤2：根据主题分布采样主题**
        
        给定文档的主题分布 $\\theta = [0.2, 0.5, 0.3]$：
        
        这意味着：
        - 50% 的概率选择主题B
        - 30% 的概率选择主题C
        - 20% 的概率选择主题A
        
        就像掷一个"加权骰子"，每个面的概率不同。
        """)
        
        # 可视化
        fig = px.bar(
            x=['主题A', '主题B', '主题C'],
            y=[0.2, 0.5, 0.3],
            color=['#3498db', '#e74c3c', '#2ecc71'],
            title="文档的主题分布 θ"
        )
        fig.update_layout(yaxis_title="概率", xaxis_title="主题")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
    
    elif gen_step == "步骤3: 生成词汇":
        st.markdown("""
        **步骤3：从主题生成词汇**
        
        每个主题都有一个词汇分布 $\\phi_k$：
        
        | 主题 | 词1 | 词2 | 词3 | 词4 |
        |------|-----|-----|-----|-----|
        | 科技 | 0.15 | 0.12 | 0.08 | ... |
        | 体育 | 0.20 | 0.15 | 0.10 | ... |
        
        如果选择了"科技"主题，就从科技主题的词分布中采样词。
        """)
    else:
        st.markdown("""
        **完整生成过程演示：**
        
        1. 📄 选择一篇文档
        2. 🎲 根据 Dirichlet(α) 生成文档的主题分布 θ
        3. 🎯 根据 θ 采样一个主题 z
        4. 📝 根据主题 z 的词分布 φ_z 采样一个词 w
        5. 🔄 重复步骤3-4，直到文档长度满足
        6. ✅ 生成的词序列就是"文档"
        """)

    st.markdown("---")
    
    # Dirichlet 分布详解
    st.subheader("📐 Dirichlet 分布详解")
    
    dir_col1, dir_col2 = st.columns([1, 1])
    
    with dir_col1:
        st.markdown("""
        ### 什么是 Dirichlet 分布？
        
        **Dirichlet 分布** 是 **Beta 分布** 的多元推广。
        
        - **Beta 分布**：生成 [0,1] 上的单个概率值
        - **Dirichlet 分布**：生成 K 个概率值，且和为1
        
        **概率密度函数**：
        
        $$P(\\theta | \\alpha) = \\frac{1}{B(\\alpha)} \\prod_{k=1}^{K} \\theta_k^{\\alpha_k - 1}$$
        
        其中 $B(\\alpha)$ 是归一化常数。
        """)
    
    with dir_col2:
        st.markdown("""
        ### Dirichlet 分布的直观理解
        
        **$\\alpha$ 参数的作用**：
        
        | $\\alpha$ 值 | 分布特点 | 直观比喻 |
        |-------------|----------|----------|
        | $\\alpha \\gg 1$ | 接近均匀分布 | 公平骰子 |
        | $\\alpha = 1$ | 均匀分布 | 无偏好 |
        | $0 < \\alpha < 1$ | 稀疏分布 | 少数值很大 |
        
        **在 LDA 中**：
        - $\\alpha$ 小 → 文档倾向于集中在少数主题
        - $\\alpha$ 大 → 文档主题分布更均匀
        """)

    st.markdown("---")
    
    # 概率图模型
    st.subheader("🔗 概率图模型")
    
    st.markdown("""
    LDA 可以用 **概率图模型** 表示，这是一种直观的可视化方式：
    
    ```
                    ┌─────────────────────────────────────┐
                    │           Plate Notation            │
                    │                                     │
                    │   ┌─────┐                          │
                    │   │  α  │  ← Dirichlet先验参数      │
                    │   └─┬───┘                          │
                    │     │                              │
                    │     ▼                              │
                    │   ┌────────┐   K个主题             │
                    │   │ θ_d   │ ← 文档主题分布         │
                    │   └───┬────┘                       │
                    │       │                            │
                    │       ▼                            │
                    │   ┌────────┐   N个词               │
                    │   │ z_d,n │ ← 词的主题分配         │
                    │   └───┬────┘                       │
                    │       │                            │
                    │       ▼                            │
                    │   ┌────────┐                       │
                    │   │ β     │ ← 主题词分布参数       │
                    │   └───┬────┘                       │
                    │       │                            │
                    │       ▼                            │
                    │   ┌────────┐                       │
                    │   │ w_d,n │ ← 观测到的词           │
                    │   └────────┘                       │
                    │                                     │
                    │   ┌─────────────┐                   │
                    │   │     M       │ ← 文档板块       │
                    │   │  docs d=1..M│                   │
                    │   └─────────────┘                   │
                    └─────────────────────────────────────┘
    ```
    """)
    
    # 板块说明
    with st.expander("📖 板块表示法说明"):
        st.markdown("""
        ### Plate Notation（板块表示法）
        
        | 符号 | 含义 |
        |------|------|
        | 圆形节点 | 随机变量 |
        | 阴影圆形 | 观测变量（已知） |
        | 方框/板块 | 重复结构 |
        | 箭头 | 依赖关系 |
        
        **板块中的数字** 表示重复次数：
        - M: 文档数量
        - N: 每篇文档的词数
        - K: 主题数量
        """)

    st.markdown("---")
    
    # 推理方法
    st.subheader("🧮 模型推理")
    
    st.markdown("""
    LDA 的 **训练/推理** 目标是找到最佳的 $\\theta$ 和 $\\phi$。
    
    由于直接计算后验分布是 ** intractable（不可行）** 的，
    我们需要使用近似推理方法：
    """)
    
    method_col1, method_col2 = st.columns([1, 1])
    
    with method_col1:
        st.markdown("""
        ### 常用推理方法
        
        | 方法 | 优点 | 缺点 |
        |------|------|------|
        | **Gibbs Sampling** | 简单可靠 | 速度较慢 |
        | **Variational Inference** | 速度快 | 近似有偏差 |
        | **LDA (sklearn)** | 使用变分推断 | 调参简单 |
        | **LDA (gensim)** | 支持Gibbs | 文档完善 |
        """)
    
    with method_col2:
        st.markdown("""
        ### 本教程使用的方法
        
        我们主要使用 **scikit-learn** 的 LDA 实现：
        
        ```python
        from sklearn.decomposition import LatentDirichletAllocation
        
        lda = LatentDirichletAllocation(
            n_components=10,      # K主题数
            doc_topic_prior=0.1,   # α
            topic_word_prior=0.01,# β
            max_iter=20           # 最大迭代
        )
        lda.fit(doc_term_matrix)
        ```
        
        sklearn 使用的是 **变分贝叶斯推断（Variational Bayes）**。
        """)

# ============================================================================
# 模块5: 文本预处理教学
# ============================================================================

def module_preprocessing():
    """文本预处理教学模块"""
    st.header("🔧 文本预处理教学")
    
    st.markdown("""
    文本预处理是 LDA 分析的关键步骤，** Garbage in, Garbage out（垃圾进，垃圾出）**。
    好的预处理能让模型学习到更有意义的主题。
    """)
    
    # 预处理流程图
    st.subheader("📋 预处理流程")
    
    st.markdown("""
    ```
    原始文本 → 分词 → 去停用词 → 词形归一 → 构建词袋 → LDA训练
       │         │          │          │          │
       ▼         ▼          ▼          ▼          ▼
    "The dogs   dogs      the       dog       {dog, run}
     running"  running  running   running    
    ```
    """)
    
    # 英文预处理演示
    st.subheader("🇬🇧 英文预处理详解")
    
    # 示例文本
    sample_english = st.text_area(
        "输入英文文本示例",
        "The machine learning algorithms are running efficiently on computers. Deep learning models process data to find patterns.",
        height=100,
        key="preprocess_english"
    )
    
    if st.button("🔄 执行预处理（英文）"):
        stop_words = load_stopwords('english_stopwords.txt')
        
        steps = []
        original = sample_english
        steps.append(("原始文本", original))
        
        # 步骤1: 转小写
        step1 = original.lower()
        steps.append(("1. 转小写", step1))
        
        # 步骤2: 去除标点和数字
        step2 = re.sub(r'[^a-zA-Z\s]', ' ', step1)
        steps.append(("2. 去除标点和数字", step2))
        
        # 步骤3: 分词
        step3 = step2.split()
        steps.append(("3. 分词", str(step3)))
        
        # 步骤4: 去停用词
        step4 = [w for w in step3 if w not in stop_words and len(w) > 2]
        steps.append(("4. 去停用词和短词", str(step4)))
        
        # 显示步骤
        for i, (title, content) in enumerate(steps):
            if i == 0:
                st.info(f"**{title}**\n\n{content}")
            else:
                prev_content = steps[i-1][1]
                st.success(f"**{title}**\n\n之前: {prev_content}\n\n之后: {content}")
    
    st.markdown("---")
    
    # 中文预处理演示
    st.subheader("🇨🇳 中文预处理详解")
    
    sample_chinese = st.text_area(
        "输入中文文本示例",
        "人工智能技术正在快速发展，机器学习和深度学习算法不断取得突破。",
        height=100,
        key="preprocess_chinese"
    )
    
    if st.button("🔄 执行预处理（中文）"):
        stop_words = load_stopwords('chinese_stopwords.txt')
        
        steps = []
        original = sample_chinese
        steps.append(("原始文本", original))
        
        # 步骤1: jieba分词
        step1 = list(jieba.cut(original))
        steps.append(("1. jieba分词", str(step1)))
        
        # 步骤2: 去停用词
        step2 = [w.strip() for w in step1 if w.strip() and w not in stop_words and len(w) > 1]
        steps.append(("2. 去停用词", str(step2)))
        
        # 显示步骤
        for i, (title, content) in enumerate(steps):
            if i == 0:
                st.info(f"**{title}**\n\n{content}")
            else:
                prev_content = steps[i-1][1]
                st.success(f"**{title}**\n\n之前: {prev_content}\n\n之后: {content}")
    
    st.markdown("---")
    
    # 预处理参数详解
    st.subheader("⚙️ 预处理参数详解")
    
    param_col1, param_col2 = st.columns([1, 1])
    
    with param_col1:
        st.markdown("""
        ### CountVectorizer 参数
        
        | 参数 | 说明 | 建议值 |
        |------|------|--------|
        | `max_df` | 忽略出现频率过高的词 | 0.7~0.95 |
        | `min_df` | 忽略出现频率过低的词 | 2~5 |
        | `max_features` | 最多保留的词数 | 1000~5000 |
        | `ngram_range` | 词元范围 | (1,1) 或 (1,2) |
        """)
    
    with param_col2:
        st.markdown("""
        ### 停用词选择原则
        
        **需要移除的词**：
        - 功能词：the, is, at, which
        - 高频无意义词：said, also, one
        - 领域无关词：需要根据场景调整
        
        **保留的词**：
        - 实义词：名词、动词、形容词
        - 领域关键词
        - 有区分度的词
        """)
    
    st.markdown("---")
    
    # 词袋模型
    st.subheader("📦 词袋模型（Bag of Words）")
    
    st.markdown("""
    **词袋模型** 是 LDA 的基础，它忽略了词的顺序，只统计词频。
    
    虽然丢失了语法和顺序信息，但这种方法：
    - ✅ 简单高效
    - ✅ 对主题分析足够有效
    - ✅ 避免了数据稀疏问题
    """)
    
    # 词袋模型示例
    bow_example = st.expander("📖 词袋模型示例")
    with bow_example:
        st.markdown("""
        ### 词袋模型转换示例
        
        **原始文档：**
        ```
        文档1: "I love machine learning"
        文档2: "I love deep learning"  
        文档3: "Machine learning is great"
        ```
        
        **构建词汇表：**
        ```
        {I, love, machine, learning, deep, is, great}
        ```
        
        **转换为词频向量：**
        ```
        文档1: [1, 1, 1, 1, 0, 0, 0]  # I, love, machine, learning
        文档2: [1, 1, 0, 1, 1, 0, 0]  # I, love, deep, learning
        文档3: [0, 0, 1, 1, 0, 1, 1]  # machine, learning, is, great
        ```
        
        **观察结果：**
        - 文档1和文档2在"love"和"learning"维度上相似
        - 文档1和文档3在"machine"和"learning"维度上相似
        - LDA会据此推断出潜在的主题结构
        """)

# ============================================================================
# 模块6: 从零运行LDA案例
# ============================================================================

def module_lda_tutorial():
    """从零运行LDA案例模块"""
    st.header("🚀 从零运行 LDA 案例")
    
    st.markdown("""
    本模块将带您完成一个完整的 LDA 主题分析流程。
    选择一个内置数据集，开始您的第一次主题分析！
    """)
    
    # 数据集选择
    st.subheader("📂 选择数据集")
    
    dataset_choice = st.selectbox(
        "选择语料库",
        ["新闻文本 (News)", "商品评论 (Reviews)", "学术摘要 (Academic)", "中文语料 (Chinese)"]
    )
    
    dataset_map = {
        "新闻文本 (News)": "news",
        "商品评论 (Reviews)": "reviews",
        "学术摘要 (Academic)": "academic",
        "中文语料 (Chinese)": "chinese"
    }
    
    # 加载数据
    data = load_sample_data(dataset_map[dataset_choice])
    
    if dataset_map[dataset_choice] == 'chinese':
        if data:
            st.success(f"✅ 加载了 {len(data)} 条中文文档")
            texts = [d['text'] for d in data]
            categories = [d['category'] for d in data]
        else:
            st.error("加载数据失败")
            return
    else:
        if not data.empty:
            st.success(f"✅ 加载了 {len(data)} 条文档")
            if 'text' in data.columns:
                texts = data['text'].tolist()
                categories = data['category'].tolist() if 'category' in data.columns else None
            else:
                st.error("数据格式不正确")
                return
        else:
            st.error("加载数据失败")
            return
    
    # 显示数据样本
    with st.expander("📋 查看数据样本"):
        if dataset_map[dataset_choice] == 'chinese':
            for i, doc in enumerate(data[:5]):
                st.markdown(f"**文档{i+1}** (类别: {doc['category']})")
                st.text(doc['text'][:200] + "...")
                st.markdown("")
        else:
            for i, row in data.head().iterrows():
                st.markdown(f"**文档{i+1}**")
                st.text(row['text'][:200] + "...")
                st.markdown("")
    
    st.markdown("---")
    
    # 参数设置
    st.subheader("⚙️ 设置 LDA 参数")
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        n_topics = st.slider("主题数量 (K)", 2, 15, 5, 1)
    with param_col2:
        max_iter = st.slider("最大迭代次数", 10, 50, 20, 5)
    with param_col3:
        alpha_param = st.select_slider(
            "Alpha (文档-主题先验)",
            options=[0.01, 0.05, 0.1, 0.5, 1.0, 'auto'],
            value=0.1
        )
    
    beta_param = st.select_slider(
        "Beta (主题-词先验)",
        options=[0.001, 0.01, 0.05, 0.1, 'auto'],
        value=0.01
    )
    
    # 预处理选项
    st.subheader("🔧 预处理选项")
    
    preprocess_col1, preprocess_col2 = st.columns([1, 1])
    
    with preprocess_col1:
        use_stopwords = st.checkbox("去除停用词", value=True)
        min_word_length = st.slider("最小词长度", 2, 5, 3, 1)
    
    with preprocess_col2:
        max_features = st.slider("最大特征数", 500, 3000, 1000, 100)
    
    # 开始训练
    if st.button("🚀 开始训练 LDA 模型", type="primary", use_container_width=True):
        with st.spinner("正在训练模型..."):
            try:
                # 预处理
                stop_words = load_stopwords('english_stopwords.txt') if dataset_map[dataset_choice] != 'chinese' else load_stopwords('chinese_stopwords.txt')
                
                if dataset_map[dataset_choice] == 'chinese':
                    processed_docs = [preprocess_chinese_text(doc, stop_words) for doc in texts]
                else:
                    processed_docs = [preprocess_english_text(doc, stop_words) for doc in texts]
                
                # 过滤空文档
                processed_docs = [doc for doc in processed_docs if doc.strip()]
                
                if len(processed_docs) < 10:
                    st.error("有效文档数量太少，请使用更多数据")
                    return
                
                # 创建词袋模型
                vectorizer = CountVectorizer(
                    max_df=0.95,
                    min_df=2,
                    max_features=max_features
                )
                
                doc_term_matrix = vectorizer.fit_transform(processed_docs)
                feature_names = vectorizer.get_feature_names_out()
                
                st.info(f"📊 词汇表大小: {len(feature_names)}")
                st.info(f"📄 文档数量: {len(processed_docs)}")
                
                # 训练 LDA
                lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    doc_topic_prior=alpha_param if alpha_param != 'auto' else None,
                    topic_word_prior=beta_param if beta_param != 'auto' else None,
                    max_iter=max_iter,
                    learning_method='online',
                    random_state=42,
                    n_jobs=-1
                )
                
                lda_model.fit(doc_term_matrix)
                
                # 获取结果
                doc_topic_dist = lda_model.transform(doc_term_matrix)
                topic_word_dist = lda_model.components_
                
                # 获取每个主题的关键词
                topics = get_top_words_per_topic(lda_model, feature_names, 10)
                
                st.success("✅ 模型训练完成！")
                
                # 显示主题
                st.markdown("---")
                st.subheader("📑 发现的主题")
                
                for topic_idx, words in topics.items():
                    with st.expander(f"📌 主题 {topic_idx + 1}"):
                        word_df = pd.DataFrame(words, columns=['关键词', '权重'])
                        word_df['排名'] = range(1, len(words) + 1)
                        word_df = word_df[['排名', '关键词', '权重']]
                        st.dataframe(word_df, use_container_width=True)
                
                # 主题词条形图
                st.markdown("---")
                st.subheader("📊 主题关键词分布")
                
                viz_topic_idx = st.selectbox("选择要查看的主题", range(n_topics), format_func=lambda x: f"主题 {x+1}")
                
                top_words = topics[viz_topic_idx]
                fig = px.bar(
                    x=[w[1] for w in top_words],
                    y=[w[0] for w in top_words],
                    orientation='h',
                    title=f"主题 {viz_topic_idx + 1} 的关键词"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # 保存模型信息到 session state
                st.session_state['lda_model'] = lda_model
                st.session_state['vectorizer'] = vectorizer
                st.session_state['doc_term_matrix'] = doc_term_matrix
                st.session_state['topics'] = topics
                st.session_state['feature_names'] = feature_names
                st.session_state['doc_topic_dist'] = doc_topic_dist
                
            except Exception as e:
                st.error(f"训练过程出错: {str(e)}")

# ============================================================================
# 模块7: 参数实验室
# ============================================================================

def module_parameter_lab():
    """参数实验室模块"""
    st.header("⚗️ 参数实验室")
    
    st.markdown("""
    在这里，您可以自由调节 LDA 的各项参数，实时观察它们对结果的影响。
    通过动手实验，深入理解每个参数的作用！
    """)
    
    # 检查是否有训练好的模型
    has_model = 'lda_model' in st.session_state
    
    if not has_model:
        st.info("💡 请先在「从零运行LDA案例」模块中训练一个模型，或使用下方快速演示")
        
        # 快速演示数据
        demo_choice = st.selectbox(
            "选择演示数据集",
            ["新闻文本", "商品评论", "学术摘要"]
        )
        
        if st.button("加载演示数据"):
            dataset_map = {"新闻文本": "news", "商品评论": "reviews", "学术摘要": "academic"}
            data = load_sample_data(dataset_map[demo_choice])
            
            if not data.empty:
                texts = data['text'].tolist()
                stop_words = load_stopwords('english_stopwords.txt')
                processed_docs = [preprocess_english_text(doc, stop_words) for doc in texts]
                processed_docs = [doc for doc in processed_docs if doc.strip()]
                
                vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=500)
                doc_term_matrix = vectorizer.fit_transform(processed_docs)
                
                # 默认训练
                lda_model = LatentDirichletAllocation(
                    n_components=5,
                    random_state=42,
                    max_iter=20
                )
                lda_model.fit(doc_term_matrix)
                
                # 保存到 session state
                st.session_state['lda_model'] = lda_model
                st.session_state['vectorizer'] = vectorizer
                st.session_state['doc_term_matrix'] = doc_term_matrix
                st.session_state['feature_names'] = vectorizer.get_feature_names_out()
                st.session_state['topics'] = get_top_words_per_topic(lda_model, st.session_state['feature_names'], 10)
                
                st.success("✅ 演示模型已加载")
                st.rerun()
            else:
                st.error("加载数据失败")
                return
    else:
        st.success("✅ 已加载训练好的模型")
    
    if has_model:
        st.markdown("---")
        
        # 参数说明
        st.subheader("📚 关键参数说明")
        
        param_info_col1, param_info_col2 = st.columns([1, 1])
        
        with param_info_col1:
            st.markdown("""
            ### K (主题数量)
            - **作用**: 控制要发现的主题数量
            - **K小**: 主题更抽象，可能混合多个概念
            - **K大**: 主题更具体，但可能过于细碎
            
            ### α (Alpha, 文档-主题先验)
            - **作用**: 控制文档主题分布的稀疏性
            - **α小**: 文档集中在少数主题
            - **α大**: 文档主题分布更均匀
            """)
        
        with param_info_col2:
            st.markdown("""
            ### β (Beta, 主题-词先验)
            - **作用**: 控制主题词分布的稀疏性
            - **β小**: 每个主题只包含少数核心词
            - **β大**: 每个主题包含更多词
            
            ### max_iter (最大迭代)
            - **作用**: 模型收敛的最大迭代次数
            - **更多迭代**: 模型更充分收敛，但收益递减
            """)
        
        st.markdown("---")
        
        # 参数实验
        st.subheader("🧪 参数实验")
        
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            exp_k = st.slider("主题数量 K", 2, 15, 5, 1, key="exp_k")
        with exp_col2:
            exp_alpha = st.select_slider(
                "Alpha",
                options=[0.01, 0.05, 0.1, 0.5, 1.0, 'auto'],
                value=0.1,
                key="exp_alpha"
            )
        with exp_col3:
            exp_beta = st.select_slider(
                "Beta",
                options=[0.001, 0.01, 0.05, 0.1, 'auto'],
                value=0.01,
                key="exp_beta"
            )
        
        if st.button("🔄 重新训练模型", use_container_width=True):
            with st.spinner("正在重新训练..."):
                try:
                    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=500)
                    doc_term_matrix = vectorizer.fit_transform(
                        [doc for doc in st.session_state.get('processed_docs', []) if doc.strip()]
                        if 'processed_docs' in st.session_state
                        else []
                    )
                    
                    if doc_term_matrix.shape[0] < 10:
                        # 使用演示数据
                        data = load_sample_data('news')
                        texts = data['text'].tolist()
                        stop_words = load_stopwords('english_stopwords.txt')
                        processed_docs = [preprocess_english_text(doc, stop_words) for doc in texts]
                        processed_docs = [doc for doc in processed_docs if doc.strip()]
                        
                        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=500)
                        doc_term_matrix = vectorizer.fit_transform(processed_docs)
                        st.session_state['processed_docs'] = processed_docs
                    
                    lda_model = LatentDirichletAllocation(
                        n_components=exp_k,
                        doc_topic_prior=exp_alpha if exp_alpha != 'auto' else None,
                        topic_word_prior=exp_beta if exp_beta != 'auto' else None,
                        max_iter=20,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    lda_model.fit(doc_term_matrix)
                    
                    st.session_state['lda_model'] = lda_model
                    st.session_state['vectorizer'] = vectorizer
                    st.session_state['doc_term_matrix'] = doc_term_matrix
                    st.session_state['feature_names'] = vectorizer.get_feature_names_out()
                    st.session_state['topics'] = get_top_words_per_topic(lda_model, st.session_state['feature_names'], 10)
                    
                    st.success("✅ 模型重新训练完成！")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"训练出错: {str(e)}")
        
        # 显示当前模型的主题
        st.markdown("---")
        st.subheader("📑 当前主题")
        
        for topic_idx, words in st.session_state.get('topics', {}).items():
            col1, col2 = st.columns([3, 1])
            with col1:
                word_str = ", ".join([f"**{w[0]}**" for w in words[:8]])
                st.markdown(f"**主题 {topic_idx + 1}**: {word_str}")
            with col2:
                st.caption(f"Top 8 词")
        
        # 困惑度随K变化图
        st.markdown("---")
        st.subheader("📈 困惑度 vs 主题数量 K")
        
        if st.button("绘制 K-困惑度曲线", use_container_width=True):
            with st.spinner("正在计算..."):
                try:
                    k_range = range(2, 11)
                    perplexities = []
                    
                    for k in k_range:
                        lda = LatentDirichletAllocation(
                            n_components=k,
                            random_state=42,
                            max_iter=10
                        )
                        lda.fit(st.session_state['doc_term_matrix'])
                        perplexities.append(lda.perplexity(st.session_state['doc_term_matrix']))
                    
                    fig = px.line(
                        x=list(k_range),
                        y=perplexities,
                        markers=True,
                        title="困惑度随主题数量K的变化"
                    )
                    fig.update_layout(xaxis_title="主题数量 K", yaxis_title="困惑度")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 建议
                    elbow_k = list(k_range)[np.argmin(perplexities)]
                    st.info(f"💡 根据困惑度，K={elbow_k} 可能是较好的选择（但需结合可解释性）")
                    
                except Exception as e:
                    st.error(f"计算失败: {str(e)}")

# ============================================================================
# 模块8: 可视化中心
# ============================================================================

def module_visualization():
    """可视化中心模块"""
    st.header("📊 可视化中心")
    
    st.markdown("""
    LDA 结果的可视化是理解主题模型的关键。本模块提供多种可视化方法，
    从不同角度展示主题结构。
    """)
    
    # 检查模型
    has_model = 'lda_model' in st.session_state
    
    if not has_model:
        st.warning("⚠️ 请先在「从零运行LDA案例」或「参数实验室」中训练模型")
        return
    
    st.success("✅ 已加载模型，开始可视化！")
    
    # 可视化类型选择
    viz_type = st.selectbox(
        "选择可视化类型",
        [
            "📊 主题关键词条形图",
            "☁️ 主题词云",
            "🗺️ 文档-主题热力图",
            "📈 主题分布饼图",
            "🔗 pyLDAvis交互图",
            "📉 困惑度曲线"
        ]
    )
    
    lda_model = st.session_state['lda_model']
    vectorizer = st.session_state['vectorizer']
    doc_term_matrix = st.session_state['doc_term_matrix']
    feature_names = st.session_state['feature_names']
    topics = st.session_state.get('topics', get_top_words_per_topic(lda_model, feature_names, 10))
    n_topics = lda_model.n_components
    
    if viz_type == "📊 主题关键词条形图":
        st.subheader("主题关键词条形图")
        
        topic_idx = st.selectbox("选择主题", range(n_topics), format_func=lambda x: f"主题 {x+1}")
        
        top_words = topics[topic_idx]
        
        fig = px.bar(
            x=[w[1] for w in top_words],
            y=[w[0] for w in top_words],
            orientation='h',
            title=f"主题 {topic_idx + 1} 的 Top 10 关键词",
            color=[w[1] for w in top_words],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "☁️ 主题词云":
        st.subheader("主题词云")
        
        topic_idx = st.selectbox("选择主题", range(n_topics), format_func=lambda x: f"主题 {x+1}")
        
        topic_word_dist = lda_model.components_
        
        wc = create_wordcloud(topic_word_dist, feature_names, topic_idx)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'主题 {topic_idx + 1} 词云', fontsize=16)
        st.pyplot(fig)
    
    elif viz_type == "🗺️ 文档-主题热力图":
        st.subheader("文档-主题热力图")
        
        doc_topic_dist = st.session_state.get('doc_topic_dist')
        if doc_topic_dist is None:
            doc_topic_dist = lda_model.transform(doc_term_matrix)
        
        # 只显示前30个文档
        n_display = min(30, doc_topic_dist.shape[0])
        
        fig = px.imshow(
            doc_topic_dist[:n_display],
            labels=dict(x="主题", y="文档", color="概率"),
            x=[f"主题{i+1}" for i in range(n_topics)],
            y=[f"文档{i+1}" for i in range(n_display)],
            color_continuous_scale='Blues',
            title=f"文档-主题分布热力图 (前{n_display}个文档)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "📈 主题分布饼图":
        st.subheader("主题分布饼图")
        
        doc_topic_dist = st.session_state.get('doc_topic_dist')
        if doc_topic_dist is None:
            doc_topic_dist = lda_model.transform(doc_term_matrix)
        
        # 计算每个主题的平均占比
        topic_means = doc_topic_dist.mean(axis=0)
        
        fig = px.pie(
            values=topic_means,
            names=[f"主题{i+1}" for i in range(n_topics)],
            title="各主题在语料库中的平均占比",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "🔗 pyLDAvis交互图":
        st.subheader("pyLDAvis 交互式可视化")
        
        st.info("💡 pyLDAvis 是最强大的 LDA 可视化工具，支持交互操作！")
        
        if st.button("生成 pyLDAvis 可视化", use_container_width=True):
            with st.spinner("正在生成可视化..."):
                try:
                    prepared = create_pyldavis_data(lda_model, vectorizer, doc_term_matrix)
                    
                    if prepared:
                        html = save_pyldavis_html(prepared)
                        
                        if html:
                            components_html = f"""
                            <iframe src="data:text/html;base64,{base64.b64encode(html.encode()).decode()}" 
                                    width="100%" height="800" style="border:none;"></iframe>
                            """
                            st.markdown("### 交互式主题图")
                            st.components.v1.html(html, height=800, scrolling=True)
                            
                            # 提供下载链接
                            st.markdown("### 📥 下载可视化")
                            st.markdown(create_download_link(html, "lda_visualization.html", "下载 HTML 文件"), unsafe_allow_html=True)
                        else:
                            st.error("无法生成 pyLDAvis HTML")
                    else:
                        st.warning("pyLDAvis 生成失败，请确保已安装 pyLDAvis 库")
                        
                except Exception as e:
                    st.error(f"可视化生成失败: {str(e)}")
        
        with st.expander("📖 pyLDAvis 使用说明"):
            st.markdown("""
            ### pyLDAvis 操作指南
            
            **左侧面板 - 主题气泡图：**
            - 每个圆圈代表一个主题
            - 圆圈大小 = 主题在语料中的占比
            - 圆圈位置 = 与其他主题的距离（相似主题靠近）
            
            **右侧面板 - 词条形图：**
            - 点击左侧某个主题，显示其关键词
            - 红色条 = 词在该主题中的概率
            - 蓝色条 = 词在所有主题中的边际概率
            
            **交互操作：**
            - 🖱️ 点击主题气泡查看关键词
            - 🔄 调整 lambda 参数（0=只关注特有词，1=只看词频率）
            """)
    
    elif viz_type == "📉 困惑度曲线":
        st.subheader("困惑度 vs 迭代次数")
        
        st.markdown("""
        困惑度（Perplexity）衡量模型对数据的拟合程度：
        - **越低越好**（理论上）
        - 但过低的困惑度可能导致过拟合
        - 需要结合主题可解释性选择
        """)
        
        # 计算不同K值的困惑度
        k_range = list(range(2, min(11, n_topics + 3)))
        
        if len(k_range) > 1:
            fig = go.Figure()
            
            perplexities = []
            for k in k_range:
                if k <= doc_term_matrix.shape[0] and k <= doc_term_matrix.shape[1]:
                    lda = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=10)
                    lda.fit(doc_term_matrix)
                    perplexities.append(lda.perplexity(doc_term_matrix))
                else:
                    perplexities.append(None)
            
            fig.add_trace(go.Scatter(
                x=k_range,
                y=perplexities,
                mode='lines+markers',
                name='困惑度'
            ))
            
            fig.update_layout(
                title='困惑度随主题数量K的变化',
                xaxis_title='主题数量 K',
                yaxis_title='困惑度'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if perplexities:
                best_k = k_range[np.nanargmin(perplexities)]
                st.info(f"💡 困惑度最低的K值: {best_k}")
        else:
            st.info("数据量不足以绘制困惑度曲线")

# ============================================================================
# 模块9: 结果解释教学
# ============================================================================

def module_result_interpretation():
    """结果解释教学模块"""
    st.header("💡 结果解释教学")
    
    st.markdown("""
    训练好 LDA 模型后，如何正确解读结果是一个重要的技能。
    本模块将教您如何阅读、分析和命名主题。
    """)
    
    # 检查模型
    has_model = 'lda_model' in st.session_state
    
    if has_model:
        lda_model = st.session_state['lda_model']
        topics = st.session_state.get('topics', {})
        n_topics = lda_model.n_components
        
        st.success("✅ 已加载模型，开始解读！")
        
        st.markdown("---")
        st.subheader("📋 当前主题概览")
        
        for topic_idx, words in topics.items():
            word_str = ", ".join([w[0] for w in words[:8]])
            with st.expander(f"主题 {topic_idx + 1}"):
                st.markdown(f"**关键词**: {word_str}")
    else:
        st.info("💡 请先训练模型，或参考以下通用解读指南")
    
    st.markdown("---")
    
    # 如何阅读主题
    st.subheader("📖 如何阅读 LDA 主题")
    
    read_col1, read_col2 = st.columns([1, 1])
    
    with read_col1:
        st.markdown("""
        ### 主题的基本结构
        
        每个主题包含：
        - **关键词列表**: 按权重排序的词
        - **权重值**: 反映词在该主题中的重要性
        
        **解读步骤**：
        1. 看前5-10个关键词
        2. 尝试总结共同主题
        3. 给主题命名
        4. 评估可解释性
        """)
    
    with read_col2:
        st.markdown("""
        ### 主题命名建议
        
        | 关键词示例 | 可能的命名 |
        |-----------|-----------|
        | player, team, game, win | 🏀 体育 |
        | algorithm, data, model | 🤖 科技 |
        | movie, actor, film | 🎬 娱乐 |
        | patient, doctor, hospital | 🏥 医疗 |
        | price, market, stock | 📈 金融 |
        """)
    
    st.markdown("---")
    
    # 常见模式
    st.subheader("🔍 常见主题模式")
    
    pattern_col1, pattern_col2 = st.columns([1, 1])
    
    with pattern_col1:
        st.markdown("""
        ### ✅ 健康的主题特征
        
        - 关键词语义相关
        - 可给出一个明确的名字
        - 与文档的实际内容相符
        - 词之间有合理的共现关系
        
        **示例**：
        ```
        主题: 科技
        关键词: computer, software, data, algorithm, 
                network, internet, digital, technology
        ```
        """)
    
    with pattern_col2:
        st.markdown("""
        ### ⚠️ 问题主题特征
        
        - 关键词语义混杂
        - 难以命名
        - 包含太多通用词
        - 主题之间高度重叠
        
        **问题诊断**：
        - K太大 → 合并相邻主题
        - 预处理不足 → 改进停用词
        - 数据不足 → 增加文档数量
        """)
    
    st.markdown("---")
    
    # 文档主题分配
    st.subheader("📄 文档主题分配")
    
    doc_topic_col1, doc_topic_col2 = st.columns([1, 1])
    
    with doc_topic_col1:
        st.markdown("""
        ### 如何解读文档的主题分布
        
        每篇文档都有一个 K 维向量，表示其在各主题上的"得分"。
        
        **示例**：
        ```
        文档A: [0.85, 0.10, 0.05]  → 主要关于主题1
        文档B: [0.40, 0.35, 0.25]  → 混合主题
        文档C: [0.05, 0.05, 0.90]  → 主要关于主题3
        ```
        """)
    
    with doc_topic_col2:
        st.markdown("""
        ### 应用建议
        
        - **文档分类**: 选择概率最高的主题作为标签
        - **相似度计算**: 用主题向量计算文档相似度
        - **聚类分析**: 基于主题分布对文档聚类
        - **趋势分析**: 跟踪主题随时间的变化
        """)
    
    # 易错提醒
    st.markdown("---")
    with st.expander("⚠️ 常见错误和注意事项"):
        st.markdown("""
        ### LDA 结果解读的常见错误
        
        | 错误 | 说明 | 正确做法 |
        |------|------|----------|
        | 过度解读单个词 | 一个词可能有多个含义 | 结合多个词一起理解 |
        | 忽视低权重词 | 重要信息可能在尾部 | 关注整体分布 |
        | 主题命名主观 | 可能偏离实际语义 | 验证于原始文档 |
        | K值选择随意 | 影响结果质量 | 使用困惑度+可解释性 |
        | 忽视迭代次数 | 模型可能未收敛 | 检查日志或增加迭代 |
        
        ### 验证主题质量的方法
        
        1. **抽样验证**: 随机抽取属于该主题的文档，检查是否符合
        2. **人工标注**: 请领域专家命名和评估
        3. **下游任务**: 在实际任务中测试效果
        4. **对比分析**: 与人工标注对比一致性
        """)

# ============================================================================
# 模块10: 模型评估与选参
# ============================================================================

def module_model_evaluation():
    """模型评估与选参模块"""
    st.header("📈 模型评估与选参")
    
    st.markdown("""
    评估 LDA 模型的质量是确保分析有效性的关键。本模块介绍常用的评估指标
    和参数选择策略。
    """)
    
    # 评估指标介绍
    st.subheader("📊 常用评估指标")
    
    metric_col1, metric_col2 = st.columns([1, 1])
    
    with metric_col1:
        st.markdown("""
        ### 困惑度 (Perplexity)
        
        **定义**: 衡量模型对未见数据的预测能力
        
        $$PP(W) = \\exp\\left(-\\frac{1}{N}\\sum_{d=1}^{D}\\log P(w_d)\\right)$$
        
        **解读**:
        - ✅ 越低越好
        - ⚠️ 但过低可能导致过拟合
        - 📌 需要与可解释性结合
        
        **scikit-learn**:
        ```python
        perplexity = model.perplexity(X_test)
        ```
        """)
    
    with metric_col2:
        st.markdown("""
        ### 一致性 (Coherence)
        
        **定义**: 衡量主题内词的语义一致性
        
        **常见计算方式**:
        - C_v: 基于滑动窗口和PMI
        - C_p: 基于条件概率
        - C_uci: 基于PMI
        - C_npmi: 归一化PMI
        
        **解读**:
        - ✅ 越高越好
        - 📌 更符合人类判断
        - ⚠️ 计算较慢
        
        **gensim**:
        ```python
        from gensim.models.coherencemodel import CoherenceModel
        coherence = CoherenceModel(model, texts, dictionary).get_coherence()
        ```
        """)
    
    st.markdown("---")
    
    # K值选择
    st.subheader("🎯 K值（主题数量）选择策略")
    
    st.markdown("""
    K值的选择是 LDA 中最关键的决策之一。以下是几种常用方法：
    """)
    
    strategy_col1, strategy_col2 = st.columns([1, 1])
    
    with strategy_col1:
        st.markdown("""
        ### 方法1: 困惑度-一致性权衡
        
        ```
        K值 ↑
          ├── 困惑度 ↓ (拟合更好)
          └── 一致性 ↓ (过拟合风险)
        
        最佳K: 找到平衡点
        ```
        
        **实践建议**:
        1. 绘制 K vs 困惑度曲线
        2. 绘制 K vs 一致性曲线
        3. 选择困惑度下降放缓且一致性较高的K
        """)
    
    with strategy_col2:
        st.markdown("""
        ### 方法2: 肘部法则
        
        类似K-Means的肘部法则：
        
        1. 计算不同K值的困惑度
        2. 绘制曲线
        3. 找到"肘部"拐点
        
        **缺点**: 肘部可能不明显
        
        ### 方法3: 主题可解释性
        
        最实用的方法：
        
        1. 尝试不同的K值
        2. 评估每个K下的主题可解释性
        3. 选择主题最有意义的K
        """)
    
    st.markdown("---")
    
    # 交互式实验
    st.subheader("🧪 K值选择实验")
    
    if 'lda_model' in st.session_state:
        doc_term_matrix = st.session_state['doc_term_matrix']
        
        if st.button("运行K值扫描", use_container_width=True):
            with st.spinner("正在扫描不同K值..."):
                k_range = range(2, 11)
                results = []
                
                for k in k_range:
                    lda = LatentDirichletAllocation(
                        n_components=k,
                        random_state=42,
                        max_iter=15
                    )
                    lda.fit(doc_term_matrix)
                    
                    perp = lda.perplexity(doc_term_matrix)
                    
                    # 简化的 coherence 计算
                    topics = get_top_words_per_topic(lda, st.session_state['feature_names'], 10)
                    
                    results.append({
                        'K': k,
                        '困惑度': perp,
                        '主题数': k
                    })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # 绘制曲线
                fig = make_subplots(rows=1, cols=2, 
                                   subplot_titles=['困惑度 vs K', '主题数 vs K'])
                
                fig.add_trace(
                    go.Scatter(x=list(k_range), y=[r['困惑度'] for r in results], 
                              mode='lines+markers', name='困惑度'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=list(k_range), y=list(k_range), name='主题数'),
                    row=1, col=2
                )
                
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **建议**: 选择困惑度开始趋于平稳，且主题可解释的K值")
    else:
        st.info("💡 请先训练模型，然后运行K值扫描")
    
    st.markdown("---")
    
    # 其他参数选择
    st.subheader("⚙️ 其他参数选择建议")
    
    param_table = {
        '参数': ['Alpha (α)', 'Beta (β)', 'max_iter', 'learning_offset'],
        '默认值': ['1/K', '1/V', '10', '1024'],
        '建议': [
            '自动或0.1',
            '自动或0.01',
            '至少20',
            '文档越多越大'
        ],
        '影响': [
            '文档主题分布稀疏性',
            '主题词分布稀疏性',
            '模型收敛程度',
            '学习稳定性'
        ]
    }
    
    st.table(pd.DataFrame(param_table))

# ============================================================================
# 模块11: LDA局限与替代
# ============================================================================

def module_alternatives():
    """LDA局限与替代方法模块"""
    st.header("⚠️ LDA 的局限与替代方法")
    
    st.markdown("""
    虽然 LDA 是最经典的主题模型，但它也有局限性。
    了解这些局限有助于在实际项目中选择合适的方法。
    """)
    
    # LDA 的局限
    st.subheader("⚠️ LDA 的主要局限")
    
    limitation_col1, limitation_col2 = st.columns([1, 1])
    
    with limitation_col1:
        st.markdown("""
        ### 技术局限
        
        | 局限 | 说明 |
        |------|------|
        | **词袋假设** | 忽略词序和上下文 |
        | **独立假设** | 词之间条件独立 |
        | **静态模型** | 不考虑时间变化 |
        | **主题数固定** | 需要预先指定K |
        | **无监督** | 无法利用先验知识 |
        """)
    
    with limitation_col2:
        st.markdown("""
        ### 实践局限
        
        | 局限 | 说明 |
        |------|------|
        | **语义问题** | 不真正理解语义 |
        | **短文本效果差** | 微博、评论等 |
        | **稀有主题难发现** | 对低频主题不敏感 |
        | **计算量大** | 大规模语料训练慢 |
        | **可解释性依赖** | 需要人工判断质量 |
        """)
    
    st.markdown("---")
    
    # 替代方法
    st.subheader("🔄 替代方法对比")
    
    methods = {
        '方法': ['LDA', 'NMF', 'LFTM', 'CTM', 'BERTopic', 'Top2Vec'],
        '类型': ['概率模型', '矩阵分解', '概率模型', '概率模型', '深度学习', '深度学习'],
        '优点': [
            '经典稳定，工具完善',
            '速度快，结果可解释',
            '支持主题相关',
            '主题可重叠',
            '语义理解强',
            '端到端，无需预处理'
        ],
        '缺点': [
            '忽略词序',
            '需要手动K',
            '计算复杂',
            '解释性较弱',
            '需要GPU',
            '新方法不够成熟'
        ],
        '适用场景': [
            '通用场景',
            '主题发现',
            '相关性分析',
            '复杂主题结构',
            '短文本/高质量主题',
            '文档嵌入'
        ]
    }
    
    st.table(pd.DataFrame(methods))
    
    st.markdown("---")
    
    # NMF 演示
    st.subheader("🧮 NMF（非负矩阵分解）演示")
    
    st.markdown("""
    **NMF (Non-negative Matrix Factorization)** 是 LDA 的重要替代方案：
    - 原理：将文档-词矩阵分解为两个非负矩阵
    - 优点：速度快，结果可解释
    - 适用：词频数据、推荐系统
    """)
    
    if 'lda_model' in st.session_state:
        doc_term_matrix = st.session_state['doc_term_matrix']
        vectorizer = st.session_state['vectorizer']
        feature_names = st.session_state['feature_names']
        
        n_topics_nmf = st.slider("选择主题数", 2, 10, 5)
        
        if st.button("运行 NMF", use_container_width=True):
            with st.spinner("正在运行 NMF..."):
                nmf_model = NMF(
                    n_components=n_topics_nmf,
                    random_state=42,
                    max_iter=500,
                    init='nndsvd'
                )
                
                nmf_model.fit(doc_term_matrix)
                
                # 获取主题词
                nmf_topics = {}
                for topic_idx, topic in enumerate(nmf_model.components_):
                    top_words_idx = topic.argsort()[:-11:-1]
                    nmf_topics[topic_idx] = [(feature_names[i], topic[i]) for i in top_words_idx]
                
                st.success("✅ NMF 模型训练完成！")
                
                # 显示结果
                for topic_idx, words in nmf_topics.items():
                    with st.expander(f"📌 NMF 主题 {topic_idx + 1}"):
                        word_df = pd.DataFrame(words, columns=['关键词', '权重'])
                        st.dataframe(word_df, use_container_width=True)
    else:
        st.info("💡 请先加载数据并训练 LDA 模型")
    
    st.markdown("---")
    
    # 方法选择指南
    st.subheader("🎯 方法选择指南")
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        ### 选择 LDA 当:
        - 通用主题分析需求
        - 需要概率解释
        - 数据量适中
        - 需要成熟工具链
        """)
    
    with guide_col2:
        st.markdown("""
        ### 选择 NMF 当:
        - 追求速度
        - 词频数据
        - 需要非负结果
        - 快速原型开发
        """)
    
    with guide_col3:
        st.markdown("""
        ### 选择 BERTopic 当:
        - 短文本为主
        - 需要高质量语义
        - 有GPU资源
        - 对主题质量要求高
        """)

# ============================================================================
# 模块12: 应用案例区
# ============================================================================

def module_applications():
    """应用案例区模块"""
    st.header("💼 应用案例区")
    
    st.markdown("""
    LDA 主题模型在实际中有广泛的应用。本模块展示几个典型的应用场景，
    并提供相应的分析模板。
    """)
    
    # 应用场景选择
    app_choice = st.selectbox(
        "选择应用场景",
        [
            "📰 新闻分类与主题追踪",
            "💬 社交媒体评论分析",
            "📚 学术文献综述",
            "🏢 企业文档管理",
            "🎯 自定义数据集分析"
        ]
    )
    
    if app_choice == "📰 新闻分类与主题追踪":
        st.subheader("📰 新闻分类与主题追踪")
        
        st.markdown("""
        ### 应用场景
        
        - **目标**: 自动识别新闻的主题类别
        - **数据**: 新闻文章、报道
        - **价值**: 
          - 快速了解新闻热点
          - 追踪主题随时间的变化
          - 自动化新闻分类
        
        ### 分析步骤
        
        1. 收集新闻文本数据
        2. 预处理（分词、去停用词）
        3. 训练 LDA 模型
        4. 分析主题分布
        5. 时间序列分析（如有时序数据）
        """)
        
        # 演示
        if st.button("使用新闻数据演示", use_container_width=True):
            with st.spinner("加载数据..."):
                data = load_sample_data('news')
                if not data.empty:
                    st.session_state['app_data'] = data
                    st.session_state['app_type'] = 'news'
                    st.success(f"✅ 加载了 {len(data)} 条新闻")
                    
                    # 显示类别分布
                    if 'category' in data.columns:
                        fig = px.histogram(data, x='category', title="新闻类别分布")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif app_choice == "💬 社交媒体评论分析":
        st.subheader("💬 社交媒体评论分析")
        
        st.markdown("""
        ### 应用场景
        
        - **目标**: 从用户评论中提取产品/服务反馈
        - **数据**: 商品评论、用户反馈
        - **价值**:
          - 了解用户满意度
          - 发现产品问题
          - 竞品分析
        
        ### 分析步骤
        
        1. 收集评论文本
        2. 预处理
        3. LDA 主题分析
        4. 结合情感分析
        5. 统计各主题的情感倾向
        """)
        
        if st.button("使用评论数据演示", use_container_width=True):
            with st.spinner("加载数据..."):
                data = load_sample_data('reviews')
                if not data.empty:
                    st.session_state['app_data'] = data
                    st.session_state['app_type'] = 'reviews'
                    st.success(f"✅ 加载了 {len(data)} 条评论")
                    
                    if 'rating' in data.columns:
                        fig = px.histogram(data, x='rating', nbins=5, title="评分分布")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif app_choice == "📚 学术文献综述":
        st.subheader("📚 学术文献综述")
        
        st.markdown("""
        ### 应用场景
        
        - **目标**: 快速了解某领域的研究主题
        - **数据**: 学术论文摘要
        - **价值**:
          - 高效文献调研
          - 发现研究趋势
          - 识别研究空白
        
        ### 分析步骤
        
        1. 收集领域论文摘要
        2. 专业预处理（保留术语）
        3. 训练 LDA 模型
        4. 主题命名与归类
        5. 分析主题演变趋势
        """)
        
        if st.button("使用学术数据演示", use_container_width=True):
            with st.spinner("加载数据..."):
                data = load_sample_data('academic')
                if not data.empty:
                    st.session_state['app_data'] = data
                    st.session_state['app_type'] = 'academic'
                    st.success(f"✅ 加载了 {len(data)} 条学术摘要")
                    
                    if 'field' in data.columns:
                        fig = px.histogram(data, x='field', title="学科分布")
                        fig.update_layout(xaxis_title="学科领域")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif app_choice == "🏢 企业文档管理":
        st.subheader("🏢 企业文档管理")
        
        st.markdown("""
        ### 应用场景
        
        - **目标**: 自动组织和管理企业文档
        - **数据**: 合同、报告、邮件
        - **价值**:
          - 智能文档分类
          - 快速信息检索
          - 知识发现
        
        ### 应用场景示例
        
        | 文档类型 | 潜在主题 |
        |---------|---------|
        | 法务合同 | 条款分析、风险识别 |
        | 财务报告 | 经营分析、预算执行 |
        | 客户邮件 | 投诉分类、需求识别 |
        | 会议纪要 | 决策跟踪、任务分配 |
        """)
    
    else:  # 自定义数据集
        st.subheader("🎯 自定义数据集分析")
        
        st.markdown("""
        ### 上传您的数据集
        
        支持的格式:
        - CSV 文件（需要有 text 列）
        - TXT 文件（每行一个文档）
        """)
        
        uploaded_file = st.file_uploader("选择文件", type=['csv', 'txt'])
        
        if uploaded_file:
            with st.spinner("加载文件..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        st.success(f"✅ 加载了 {len(df)} 条记录")
                        
                        # 显示列名
                        st.write("可用列:", df.columns.tolist())
                        
                        # 选择文本列
                        text_col = st.selectbox("选择文本列", df.columns)
                        
                        if st.button("开始分析", use_container_width=True):
                            st.session_state['app_data'] = df
                            st.session_state['app_texts'] = df[text_col].tolist()
                            st.session_state['app_type'] = 'custom'
                            st.success("✅ 数据准备完成，请在其他模块进行分析")
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        docs = [line.strip() for line in content.split('\n') if line.strip()]
                        st.session_state['app_texts'] = docs
                        st.session_state['app_type'] = 'custom'
                        st.success(f"✅ 加载了 {len(docs)} 条文档")
                        
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
    
    st.markdown("---")
    
    # 最佳实践
    with st.expander("📋 应用最佳实践"):
        st.markdown("""
        ### LDA 应用最佳实践
        
        | 阶段 | 建议 |
        |------|------|
        | **数据收集** | 数据量至少100+文档，主题覆盖要全 |
        | **预处理** | 根据领域调整停用词表，保留专业术语 |
        | **参数选择** | K值5-20较常用，alpha/beta自动即可 |
        | **结果验证** | 抽样检查主题是否可解释 |
        | **迭代优化** | 根据结果调整预处理和参数 |
        | **业务结合** | 结合领域知识命名主题 |
        
        ### 常见问题处理
        
        - **主题重叠严重**: 减少K或调整alpha
        - **主题难以解释**: 增加预处理或增加迭代
        - **短文本效果差**: 减少停用词或使用Word2Vec增强
        """)

# ============================================================================
# 模块13: 练习与自测
# ============================================================================

def module_quiz():
    """练习与自测模块"""
    st.header("📝 练习与自测")
    
    st.markdown("""
    通过以下练习题，检验您对 LDA 主题模型的理解程度！
    """)
    
    # 练习题
    questions = [
        {
            "question": "LDA 的全称是什么？",
            "options": [
                "A. Latent Dirichlet Allocation（潜在狄利克雷分配）",
                "B. Linear Discriminant Analysis（线性判别分析）",
                "C. Latent Semantic Analysis（潜在语义分析）",
                "D. Latent Dirichlet Algorithm（潜在狄利克雷算法）"
            ],
            "answer": 0,
            "explanation": "LDA = Latent Dirichlet Allocation，中文译为潜在狄利克雷分配"
        },
        {
            "question": "在 LDA 中，α（Alpha）参数控制什么？",
            "options": [
                "A. 主题数量",
                "B. 文档-主题分布的稀疏性",
                "C. 词袋大小",
                "D. 迭代次数"
            ],
            "answer": 1,
            "explanation": "Alpha 控制文档-主题分布的先验参数，影响每个文档的主题分布稀疏性"
        },
        {
            "question": "LDA 属于哪种类型的模型？",
            "options": [
                "A. 监督学习模型",
                "B. 半监督学习模型",
                "C. 无监督学习模型",
                "D. 强化学习模型"
            ],
            "answer": 2,
            "explanation": "LDA 是一种无监督学习方法，不需要人工标注数据"
        },
        {
            "question": "以下哪个不是 LDA 的局限性？",
            "options": [
                "A. 忽略词的顺序",
                "B. 需要预先指定主题数K",
                "C. 能够理解语义",
                "D. 短文本效果较差"
            ],
            "answer": 2,
            "explanation": "LDA 不能真正理解语义，它只基于词的共现统计"
        },
        {
            "question": "在 LDA 中，一篇文档可以属于多少个主题？",
            "options": [
                "A. 只能一个",
                "B. 两个",
                "C. 三个",
                "D. 可以是多个（软聚类）"
            ],
            "answer": 3,
            "explanation": "LDA 支持软聚类，一篇文档可以同时属于多个主题，概率之和为1"
        },
        {
            "question": "困惑度（Perplexity）越低意味着什么？",
            "options": [
                "A. 模型越好（一定能用）",
                "B. 模型拟合越好（但可能过拟合）",
                "C. 主题数越多越好",
                "D. 数据量越大越好"
            ],
            "answer": 1,
            "explanation": "困惑度越低表示模型拟合越好，但过低的困惑度可能表示过拟合"
        },
        {
            "question": "LDA 的「生成过程」是什么顺序？",
            "options": [
                "A. 采样词 → 采样主题 → 采样文档分布",
                "B. 采样文档分布 → 采样主题 → 采样词",
                "C. 采样主题 → 采样文档分布 → 采样词",
                "D. 采样词 → 采样文档分布 → 采样主题"
            ],
            "answer": 1,
            "explanation": "生成过程：先采样文档的主题分布，再根据分布采样主题，最后从主题采样词"
        },
        {
            "question": "主题一致性（Coherence）衡量什么？",
            "options": [
                "A. 模型训练速度",
                "B. 主题内词的语义相关性",
                "C. 文档数量",
                "D. 词汇表大小"
            ],
            "answer": 1,
            "explanation": "Coherence 衡量主题内词汇的语义一致性和共现程度"
        }
    ]
    
    # 存储答案
    if 'quiz_answers' not in st.session_state:
        st.session_state['quiz_answers'] = {}
        st.session_state['quiz_submitted'] = False
    
    # 显示题目
    for i, q in enumerate(questions):
        st.markdown(f"### 题目 {i+1}")
        st.markdown(q['question'])
        
        selected = st.radio(
            f"选择答案 ({i+1})",
            q['options'],
            index=None,
            key=f"q_{i}"
        )
        
        if selected:
            st.session_state['quiz_answers'][i] = q['options'].index(selected)
    
    st.markdown("---")
    
    # 提交和评分
    if st.button("📤 提交答案", use_container_width=True):
        st.session_state['quiz_submitted'] = True
    
    if st.session_state['quiz_submitted']:
        score = 0
        for i, q in enumerate(questions):
            if st.session_state['quiz_answers'].get(i) == q['answer']:
                score += 1
        
        st.markdown(f"## 📊 得分: {score}/{len(questions)}")
        
        # 显示详细结果
        for i, q in enumerate(questions):
            user_answer = st.session_state['quiz_answers'].get(i)
            
            if user_answer == q['answer']:
                st.success(f"✅ 题目 {i+1}: 正确！")
            else:
                st.error(f"❌ 题目 {i+1}: 错误！")
                st.markdown(f"正确答案是: {q['options'][q['answer']]}")
            
            st.info(f"💡 解释: {q['explanation']}")
            st.markdown("")
    
    # 附加练习
    st.markdown("---")
    st.subheader("🎯 附加练习")
    
    with st.expander("💻 编程练习"):
        st.markdown("""
        ### 练习1: 基础 LDA 实现
        
        使用 scikit-learn 实现 LDA 主题分析：
        
        ```python
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        
        # 1. 准备数据
        documents = ["你的文档列表"]
        
        # 2. 构建词袋
        vectorizer = CountVectorizer(max_df=0.95, min_df=2)
        X = vectorizer.fit_transform(documents)
        
        # 3. 训练 LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        
        # 4. 查看主题
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            print(f"主题 {topic_idx}: {top_words}")
        ```
        
        **任务**:
        1. 加载内置数据集
        2. 设置 K=5
        3. 提取每个主题的 Top 10 关键词
        4. 给每个主题命名
        """)
    
    # 学习资源
    st.markdown("---")
    st.subheader("📚 学习资源")
    
    resource_col1, resource_col2 = st.columns([1, 1])
    
    with resource_col1:
        st.markdown("""
        ### 推荐阅读
        
        - **原论文**: Latent Dirichlet Allocation (Blei et al., 2003)
        - **通俗解释**: Probabilistic Topic Models (Blei, 2012)
        - **scikit-learn 文档**: LDA 实现说明
        """)
    
    with resource_col2:
        st.markdown("""
        ### 进阶主题
        
        - Gibbs Sampling for LDA
        - Dynamic Topic Models
        - Correlated Topic Models
        - Supervised LDA
        - BERTopic 等深度学习方法
        """)

# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """
    主函数 - 程序的入口点
    
    功能说明:
    1. 渲染页面标题和侧边栏
    2. 根据选择的模块显示对应内容
    3. 处理模块间的状态共享
    """
    
    # 初始化 session state
    if 'current_module' not in st.session_state:
        st.session_state['current_module'] = 'welcome'
    
    # 渲染页面标题
    render_header()
    
    # 渲染侧边栏并获取选择的模块
    selected_module = render_sidebar()
    st.session_state['current_module'] = selected_module
    
    # 根据选择的模块显示对应内容
    if selected_module == "welcome":
        module_welcome()
    elif selected_module == "what_is_topic":
        module_what_is_topic()
    elif selected_module == "lda_intuition":
        module_lda_intuition()
    elif selected_module == "math_model":
        module_math_model()
    elif selected_module == "preprocessing":
        module_preprocessing()
    elif selected_module == "lda_tutorial":
        module_lda_tutorial()
    elif selected_module == "parameter_lab":
        module_parameter_lab()
    elif selected_module == "visualization":
        module_visualization()
    elif selected_module == "result_interpretation":
        module_result_interpretation()
    elif selected_module == "model_evaluation":
        module_model_evaluation()
    elif selected_module == "alternatives":
        module_alternatives()
    elif selected_module == "applications":
        module_applications()
    elif selected_module == "quiz":
        module_quiz()
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>📚 LDA 文本主题分析教学演示 | 版本 v1.0</p>
        <p>帮助您从原理到实践全面掌握 LDA 主题模型</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 程序入口点
# ============================================================================

if __name__ == "__main__":
    main()
