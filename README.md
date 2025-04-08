# Financial Fraud Detection using Text and Structured Data

本项目旨在结合财务数据、违规记录与管理层讨论与分析（MD&A）文本信息，构建一个预测财务舞弊行为的模型。通过BERT提取语义特征，并融合结构化数据进行多模态建模，提升识别精度。

---

## 项目结构

```plaintext
├── data_clean.py        # 原始数据的清洗与标准化处理 
├── get_bert.py          # 提取MD&A文本的BERT特征 
├── merge_data.py        # 合并财务数据、违规记录与文本特征 
├── check_data.py        # 检查样本分布（正负类比例） 
├── model_share.py       # 模型训练脚本（当前版本尚待优化） 
├── requirements.txt     # 项目依赖库文件
```
---

## 各模块说明

### `data_clean.py`
对原始数据集（如财务报表、企业违规记录等）进行预处理，包括去重、缺失值处理、统一格式等，为后续处理打基础。

### `get_bert.py`
使用BERT模型对MD&A文本进行特征提取，输出每份文本的向量表示。支持本地BERT模型或API方式加载。

### `merge_data.py`
将清洗后的财务数据、违规信息与文本特征整合为统一表格，用于模型输入。

### `check_data.py`
分析合并后数据的类别分布情况，输出每年正类（舞弊）与负类（非舞弊）样本数量，辅助判断数据是否需要重采样。

### `model_share.py`
融合结构化特征与BERT文本向量的MLP模型。采用SMOTE过采样处理类别不平衡，支持EarlyStopping与超参数调优（基于Keras Tuner）。当前模型仍在持续迭代优化中。

---

## 运行流程

建议依照以下顺序执行脚本：

1. 运行 `data_clean.py` 对原始数据进行清洗；
2. 运行 `get_bert.py` 提取MD&A文本特征；
3. 运行 `merge_data.py` 合并所有特征数据；
4. 运行 `check_data.py` 查看类别分布，决定是否进行重采样；
5. 运行 `model_share.py` 开始模型训练与评估。

---

## 环境依赖

### 使用 `requirements.txt` 文件安装依赖库

