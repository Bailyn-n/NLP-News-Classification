# NLP 新闻文本分类实战 (Datawhale 实践项目)

这是一个基于天池“零基础入门NLP”赛题的深度学习实践项目。目标是处理 20 万条匿名化处理后的新闻文本，将其准确分类到 14 个候选类别中（如科技、股票、体育等）。

## 1. 项目背景与目标
- **赛题背景**：赛题数据已进行字符级别匿名化处理，无法直接使用预训练的中文词向量。这要求开发者从零开始进行特征提取和模型构建。
- **核心挑战**：
    - 匿名字符的语义建模。
    - 类别不平衡问题（某些类别样本极少）。
    - 长文本的截断与信息保留。
- **技术栈**：Python, Pandas, Scikit-learn, FastText, PyTorch。

## 2. 环境配置 (Apple Silicon M3 优化)
本项目在 **MacBook Air M3** 上开发，并针对 Apple Silicon 进行了性能优化。

- **操作系统**: macOS
- **开发环境**: VS Code + Jupyter Notebook
- **虚拟环境管理**: Conda (nlp_env)
- **主要依赖库**:
    - `python 3.10+`
    - `pandas`: 数据处理与分析
    - `scikit-learn`: 传统机器学习模型与评价指标
    - `fasttext`: 深度学习快速文本分类
    - `numpy < 2.0`: (针对 fasttext 的兼容性降级处理)
- **硬件加速**: 
    - 深度学习部分通过 **MPS (Metal Performance Shaders)** 实现 M3 芯片的 GPU 加速。

## 3. 当前进度
- [x] Task 1: 赛题理解与环境搭建
- [x] Task 2: 数据读取与 EDA 分析
- [x] Task 3: 特征提取与机器学习模型 (Baseline: 0.904)
- [x] Task 4: 深度学习模型 (FastText)
    - **实验 A：手动控制变量**
        - 验证了 `wordNgrams` 的影响：ngram=1 (0.8432) -> ngram=2 (0.8851) -> ngram=3 (**0.8868**)。
    - **实验 B：自动调参 (Autotune)**
        - 设置 `autotuneDuration=600`，历经 6 次试验 (Trials)。
        - 最佳参数：lr=1.768, epoch=8, dim=313, wordNgrams=1。
        - 最终 F1 分数：**0.8886**。
    - **R&D 洞察**：
        - 自动化调参能挖掘出人工难以直观组合的参数（如极高的学习率配合高维向量）。
        - 在有限时间内，Autotune 通过权衡 epoch 和 dim，找到了比单纯增加 ngram 更好的全局最优解。
- [ ] Task 5: 深度学习模型 (Word2Vec + CNN/RNN) (进行中...)