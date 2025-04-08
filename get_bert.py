import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import os


os.environ["HF_TOKEN"] = yours


# 文本数据
input_path = "./data/管理层讨论与分析_cleaned.csv"
df = pd.read_csv(input_path, encoding='utf-8')


# 统计每个公司出现的次数
company_counts = df['公司简称'].value_counts()

invalid_companies = company_counts[company_counts < 10].index

print(invalid_companies)
print(len(invalid_companies))


# 检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载BERT
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)


def extract_bert_features(text):
    """ 提取文本的 BERT [CLS] 向量 """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # 发送到 GPU
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 送回 CPU 处理
    return cls_embedding


# 处理所有文本，加入进度显示
features_list = []
for idx, text in enumerate(df['经营讨论与分析内容'].astype(str)):
    features_list.append(extract_bert_features(text))
    
    # 每处理 1000 条数据，打印进度
    if (idx + 1) % 1000 == 0:
        print(f"已处理 {idx + 1} 条文本，共 {len(df)} 条")

# 转换为numpy数组
X = np.vstack(features_list)

# 索引对齐
df_bert = pd.DataFrame(X, columns=[f'bert_feature_{i}' for i in range(X.shape[1])])
df_bert.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

df_final = pd.concat([df.drop(columns=['经营讨论与分析内容']), df_bert], axis=1)

# 保存
output_path = "./data/管理层讨论与分析_bert.csv"
df_final.to_csv(output_path, index=False, encoding='utf-8')

print(f"数据已保存至 {output_path}")