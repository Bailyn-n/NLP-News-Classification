## 当前进度
- [x] Task 1: 赛题理解与环境搭建
- [x] Task 2: 数据读取与 EDA 分析
- [x] Task 3: 特征提取与机器学习模型 (Baseline 搭建)
    - 使用 **TF-IDF** 进行了文本向量化（ngram_range=1-3, features=5000）。
    - 对比了 **RidgeClassifier** 和 **LogisticRegression** 模型。
    - 实验发现增加数据量（从 1.5w 到 5w）能显著提升 F1 分数（从 0.87 提升至 **0.904**）。
    - 验证了线性模型在处理高维稀疏特征时的稳健性。
- [ ] Task 4: 深度学习模型 (Word2Vec + CNN/RNN) (进行中...)