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
- [x] Task 5: 深度学习模型进阶 (Word2Vec + 神经网络)
    - **作业 1：Word2Vec 词向量训练**
        - 使用 **Skip-gram** 模式对 10 万条数据进行了词向量建模。
        - 成功将匿名数字映射到 100 维的稠密空间。
        - **实验发现**：数字 '3750' 与 '900'、'648' 等具有极高的余弦相似度（>0.95），验证了匿名字符之间存在显著的语义/语法关联，为后续 CNN/RNN 建模打下了地基。
- [x] Task 5: 深度学习模型进阶 (Word2Vec + TextCNN/TextRNN/HAN)
    - **预训练嵌入**：使用 Gensim 构建 Word2Vec 模型，生成 100 维词向量。
    - **架构对比实验 (基于 5 万条训练数据, Epoch=2)**：
        - **TextCNN**：耗时 36 分钟，Loss 0.37。通过多尺度卷积核高效提取 n-gram 特征。
        - **TextRNN (Bi-LSTM)**：耗时 4 小时，Loss 0.85。串行序列过长（200词）导致计算极慢且存在信息遗忘。
        - **HAN (Hierarchical Attention Network)**：**耗时仅 51 秒，Loss 降至 0.30（全场最佳）！** 
    - **R&D 深度优化总结**：
        在 HAN 模型中，我引入了**序列重塑（Sequence Reshaping）策略**，将 200 词长序列折叠为 10 句 × 20 词。这不仅让 Attention 机制能够分别在词级和句级精准“划重点”过滤噪声（达成最低 Loss），更将时间复杂度的瓶颈打破，利用并行化将训练速度相较于原生 RNN 提升了近 280 倍！深刻验证了注意力机制在 NLP 任务中的绝对统治力。
- [x] Task 6: 基于深度学习的文本分类 3 (BERT Pretrain + Finetune)
    - **数据特点与建模策略**：赛题文本为匿名数字 token 序列（非原始中文），无法直接套用现成中文 BERT tokenizer。因此采用**自建词表 + 小型 BERT**的方式完成预训练与微调。
    - **自定义 BERT 词表与输入管线**：
        - 基于训练集统计得到 **6869 个普通 token**，加入 `[PAD]`、`[UNK]`、`[CLS]`、`[SEP]`、`[MASK]` 五类特殊 token，最终 `vocab_size=6874`。
        - 构建了适配匿名 token 序列的 `NewsMLMDataset` 与 `NewsClsDataset`，完成了 BERT 风格输入封装（`[CLS] ... [SEP]`）、padding、`attention_mask`、以及 MLM masking/label 生成。
    - **作业 1：BERT 预训练 (MLM Pretrain)**
        - 使用 `BertConfig` 从零初始化小型 BERT：`hidden_size=256`、`num_hidden_layers=4`、`num_attention_heads=4`、`intermediate_size=1024`、`max_position_embeddings=256`。
        - 在 2000 条样本子集上进行了 1 个 epoch 的 Masked Language Modeling 预训练，训练 loss 从 **8.8774** 下降至 **6.x** 区间，最终 `epoch avg loss = 6.8589`。
        - 成功保存预训练 checkpoint：`saved_models/bert_mlm_small`。
    - **作业 2：BERT 微调 (Sequence Classification Finetune)**
        - 基于相同 BERT 主体结构构建 `BertForSequenceClassification(num_labels=14)`，并加载 MLM 预训练得到的 encoder 参数进行迁移学习。
        - 首轮微调实验结果：`avg_train_loss=2.2394`、`dev_loss=1.8676`、`dev_acc=0.3350`，显著高于 14 分类随机猜测水平（约 `1/14≈0.071`），验证了 **Pretrain + Finetune** 流程在匿名文本上的有效性。
    - **R&D 阶段性结论**：
        - 对于无法直接复用中文预训练模型的匿名数字语料，**自定义词表 + 小型 BERT + MLM 预训练**是一条可行路线。
        - 当前结果已完整打通 `原始匿名 token → MLM 预训练 → 分类微调` 的工程闭环，为后续开展 `pretrain vs no-pretrain` 对照实验、以及 `max_len / learning_rate / batch_size` 调参提供了稳定基线。




## 4. 核心实验与模型演进 (R&D 记录)
### Task 4 最终调参实验：学习率 (lr) 敏感度测试

在固定 `wordNgrams=2` 和 `epoch=25` 的前提下，我对学习率进行了网格搜索，结果如下：

| 学习率 (lr) | F1 Score (Macro) | 状态 |
| :--- | :--- | :--- |
| 0.1 | 0.8510 | 欠拟合 (学习不足) |
| **0.5** | **0.8827** | **最优平衡点** |
| 0.8 | 0.8808 | 学习率过高 (开始震荡) |
| 1.2 | 0.8806 | 学习率过高 (性能饱和) |

**研发结论**：
实验证明，对于本赛题的匿名数字特征，学习率在 0.5 附近具有最佳的收敛效果。虽然 Autotune 曾给出更高学习率的建议，但在手动控制维度（dim=100）的情况下，0.5 是目前性价比最高的参数。

### Task 5: 深度学习模型进阶 (Word2Vec + 神经网络)

本环节彻底摒弃了传统的机器学习特征工程，转而使用 PyTorch 从零搭建深度学习管线，验证了三种经典深度学习架构在匿名文本上的表现。

**1. 作业 1：Word2Vec 词向量预训练**
- 使用 `Gensim` 库训练了 Word2Vec 模型（Skip-gram 模式，维度 100，窗口 5）。
- **验证成果**：成功将离散的匿名数字映射到稠密的向量空间。实验发现，高频字符（如 '3750'、'900'、'648'，推测为标点符号或停用词）的相互余弦相似度极高。这证明模型有效捕捉到了字符间的语义与语法关联，为后续神经网络提供了高质量的初始 Embedding 权重。

**2. 作业 2 & 3：神经网络架构对比实验 (TextCNN / TextRNN / HAN)**
在使用统一的预训练 Embedding、5 万条训练数据、均训练 2 个 Epoch 的严控变量条件下，模型表现差异巨大：

| 模型架构 | 核心机制 | 训练总耗时 | 最终 Loss | 状态评估 |
| :--- | :--- | :--- | :--- | :--- |
| **TextCNN** | 卷积核并行提取 n-gram 局部特征 | 约 36 分钟 | 0.3715 | 收敛快，并行计算效率高 |
| **TextRNN** | 双向 LSTM 提取时序逻辑特征 | 约 4 小时 (14324s) | 0.8476 | 串行耗时长，存在长序列信息遗忘瓶颈 |
| **HAN** | 层级注意力机制 (词级 + 句级) | **仅 51.51 秒** | **0.3062** | **速度与精度全场最佳** |

**💡 研发结论与洞察 (R&D Insight)**：
1. **模型效率与算力匹配**：传统的 TextRNN 在处理长度为 200 的长序列时，由于必须等待上一个时间步完成（串行计算），极大地限制了 Apple M3 芯片的多核并发性能，导致耗时高达 4 小时。
2. **特征偏好**：TextCNN 的收敛速度和 Loss 表现远好于 TextRNN，说明对于新闻主题分类任务，寻找“局部核心词组”比“通读全局逻辑”更有效。
3. **注意力机制的降维打击**：在 HAN 模型中，我引入了 **序列重塑（Sequence Reshaping）** 策略，将单篇 200 词长序列物理折叠为 10 句 × 20 词的矩阵。这一工程学改造不仅打破了 RNN 的时序计算瓶颈（提速近 280 倍），更让 Attention 机制能够精准“划重点”，自动赋予停用词极低的权重，取得了 0.30 的最低 Loss。这深刻印证了 Attention 机制在现代 NLP 模型中的统治地位。

### Task 6: 基于深度学习的文本分类 3 (BERT Pretrain + Finetune)

本环节针对赛题“匿名数字 token 序列”的数据特点，放弃直接使用现成中文 BERT tokenizer，转而采用**自定义词表 + 小型 BERT + MLM 预训练 + 分类微调**的完整流程，以验证 Transformer 架构在匿名文本分类任务中的可行性。

**1. 作业 1：BERT 预训练 (Masked Language Modeling)**
- **词表构建**：统计训练集得到 6869 个普通 token，并加入 `[PAD]`、`[UNK]`、`[CLS]`、`[SEP]`、`[MASK]` 五类特殊 token，最终 `vocab_size=6874`。
- **输入建模**：自定义 `NewsMLMDataset`，将样本统一封装为 `[CLS] token_1 token_2 ... token_n [SEP]`，并自动生成 `attention_mask`、MLM 的 masked input 与 labels。
- **模型配置**：使用 `BertConfig` 从零初始化轻量级 BERT：
  - `hidden_size=256`
  - `num_hidden_layers=4`
  - `num_attention_heads=4`
  - `intermediate_size=1024`
  - `max_position_embeddings=256`
- **训练结果**：在 2000 条样本上完成 1 个 epoch 的 smoke test，loss 从 **8.8774** 稳定下降至 **6.x**，最终 `epoch avg loss = 6.8589`，证明 MLM 训练信号有效。
- **模型保存**：成功保存预训练 checkpoint：`saved_models/bert_mlm_small`。

**2. 作业 2：BERT 微调 (Sequence Classification)**
- 基于相同配置构建 `BertForSequenceClassification(num_labels=14)`。
- 将 MLM 预训练得到的 BERT encoder 参数迁移至分类模型，仅保留 pooler 与 classifier 为随机初始化。
- 补充了统一分类评估函数，使用：
  - `dev_loss`
  - `accuracy`
  - `macro_f1`

**3. 对照实验：Pretrain vs No-Pretrain**
在相同分类结构与训练流程下，对比了“先进行 MLM 预训练再微调”和“直接随机初始化分类模型进行训练”两种方案：

| Setting | avg_train_loss | dev_loss | dev_acc | dev_macro_f1 |
| :-- | --: | --: | --: | --: |
| **Pretrain + Finetune** | 1.7370 | 1.5291 | 0.5550 | 0.1943 |
| **Direct Finetune (No Pretrain)** | 2.1120 | 1.5643 | 0.5850 | 0.2034 |

**4. 当前阶段结论**
- 目前已经完整打通 `自建 vocab → MLM pretrain → classification finetune → dev evaluation` 的工程闭环。
- 从当前这一次小规模实验结果来看，**No-Pretrain 略优于 Pretrain**：
  - `dev_acc: 0.5850 > 0.5550`
  - `dev_macro_f1: 0.2034 > 0.1943`
- 这说明在当前设置下，**轻量级 MLM 预训练尚未带来稳定的下游收益**。
- 一个更合理的解释不是“预训练无效”，而是：
  1. 预训练规模过小（仅 2000 条样本，1 个 epoch）
  2. 小型 BERT 的表示容量有限
  3. 当前微调学习率和训练轮数可能尚未与预训练权重充分匹配

**💡 R&D Insight**
1. **匿名数字语料也可以做 BERT**：虽然无法直接复用现成中文 tokenizer，但通过自定义词表和输入管线，仍然可以成功完成 BERT 的预训练与微调。
2. **本阶段重点是打通闭环，而非追求极致精度**：当前最重要的成果，是验证了匿名 token 序列上的 BERT 工程路径可行。
3. **下一阶段优化方向已经明确**：
   - 扩大 MLM 预训练规模
   - 调整 finetune 学习率
   - 比较 `max_len / batch_size / learning_rate`
   - 进一步验证 `pretrain vs no-pretrain` 的稳定性



