# LDA 文本主题分析教学演示

## 项目简介

这是一个专为教学设计的 LDA（Latent Dirichlet Allocation，潜在狄利克雷分配）主题模型交互式学习工具。通过丰富的可视化、逐步演示和交互式实验，帮助学习者从原理到实践全面掌握 LDA 文本主题分析技术。

## 功能特性

- 📚 **13个核心教学模块**：从基础概念到高级应用，系统化学习路径
- 🎨 **丰富的可视化**：词云、热力图、pyLDAvis交互图等
- ⚡ **实时参数实验**：动态调整K值、alpha、beta等参数并观察效果
- 📝 **详细代码注释**：每段代码都有教学级别的注释说明
- 🌐 **内置多个语料库**：新闻、评论、学术摘要、中文小语料

## 安装

### 方式一：使用 pip 安装（推荐）

```bash
cd LDA教学演示
pip install -r requirements.txt
```

### 方式二：使用 conda 环境

```bash
conda create -n lda_teaching python=3.10
conda activate lda_teaching
pip install -r requirements.txt
```

## 运行

安装完成后，在项目目录下运行：

```bash
cd LDA教学演示
streamlit run app.py
```

浏览器将自动打开，默认地址：`http://localhost:8501`

## 项目结构

```
LDA教学演示/
├── app.py              # 主程序（所有功能集成）
├── requirements.txt    # 依赖清单
├── README.md          # 本说明文件
├── data/              # 内置语料库
│   ├── news_sample.csv
│   ├── reviews_sample.csv
│   ├── academic_sample.csv
│   └── chinese_sample.txt
└── stopwords/         # 停用词表
    ├── english_stopwords.txt
    └── chinese_stopwords.txt
```

## 教学模块说明

| 模块 | 内容 |
|------|------|
| 1. 欢迎页 | 项目介绍、学习路径导航 |
| 2. 什么是主题模型 | 目标、概念、与TF-IDF等方法对比 |
| 3. LDA直觉理解 | 调色盘/菜谱比喻、动画演示 |
| 4. 数学与概率图模型 | 变量定义、生成过程、Dirichlet分布 |
| 5. 文本预处理教学 | 分词、去停用词、词袋模型 |
| 6. 从零运行LDA | 完整案例演示 |
| 7. 参数实验室 | K值、alpha、beta调节 |
| 8. 可视化中心 | 各种可视化图表 |
| 9. 结果解释教学 | 如何阅读主题、命名建议 |
| 10. 模型评估 | 困惑度、一致性、K值选择 |
| 11. LDA局限与替代 | NMF、BERTopic等对比 |
| 12. 应用案例区 | 多领域实际应用 |
| 13. 练习与自测 | 交互式练习题 |

## 内置数据集

### 英文语料库
- **新闻文本** (news_sample.csv)：来自20newsgroups的子集
- **商品评论** (reviews_sample.csv)：产品评论文本
- **学术摘要** (academic_sample.csv)：学术论文摘要

### 中文语料库
- **中文小语料** (chinese_sample.txt)：新闻类中文短文本

## 使用技巧

1. **新手建议**：按模块顺序学习，从欢迎页开始
2. **参数实验**：在参数实验室模块多尝试不同参数组合
3. **代码学习**：查看各模块的详细注释，理解每一步
4. **自定义数据**：支持上传自己的文本数据进行主题分析

## 常见问题

### Q: 提示缺少某个包？
请确保已安装 requirements.txt 中的所有依赖：
```bash
pip install -r requirements.txt
```

### Q: 中文显示乱码？
程序会自动配置中文字体支持，如仍有问题请检查系统字体。

### Q: pyLDAvis 无法显示？
pyLDAvis 需要浏览器支持 WebGL，请确保使用现代浏览器。

## 学习路径建议

```
入门路线（建议2-3小时）：
1. 模块1-3：概念入门（约30分钟）
2. 模块4-5：原理理解（约45分钟）
3. 模块6-7：动手实践（约45分钟）
4. 模块8-9：结果解读（约30分钟）
5. 模块10-13：深入拓展（自由安排）
```

## 贡献与反馈

如果您发现任何问题或有改进建议，欢迎反馈！

## 许可证

MIT License

---

**制作目的**：让更多人能够轻松理解并应用 LDA 主题模型
**版本**：v1.0
**更新日期**：2024年
