import pandas as pd
import matplotlib.pyplot as plt

# merged为合并后的数据，经过填充0处理
merged_df = pd.read_csv('merged_data.csv')

merged_df['会计年度'] = pd.to_numeric(merged_df['会计年度'], errors='coerce')
merged_df['是否违规'] = pd.to_numeric(merged_df['是否违规'], errors='coerce')

# 按会计年度和是否违规进行分组计数
yearly_counts = merged_df.groupby('会计年度')['是否违规'].value_counts().unstack(fill_value=0)
print("每年正负样本数量：")
print(yearly_counts)

# 如果数据中没有正负行中缺少某个类别，可以用以下代码确保两列均存在
if 0 not in yearly_counts.columns:
    yearly_counts[0] = 0
if 1 not in yearly_counts.columns:
    yearly_counts[1] = 0

# 计算正样本（违规）的比例
yearly_counts['Fraud Ratio'] = yearly_counts[1] / (yearly_counts[0] + yearly_counts[1])
print("Fraud Ratio per Fiscal Year:")
print(yearly_counts[['Fraud Ratio']])

# 绘制每年的正负样本数量堆积柱状图
yearly_counts[[0, 1]].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Fraud vs Non-Fraud Counts per Fiscal Year")
plt.xlabel("Fiscal Year")
plt.ylabel("Number of Samples")
plt.legend(["Non-Fraud (0)", "Fraud (1)"])
plt.show()

# 绘制每年的正样本比例折线图
plt.figure(figsize=(10, 6))
plt.plot(yearly_counts.index, yearly_counts['Fraud Ratio'], marker='o', linestyle='-')
plt.title("Fraud Ratio per Fiscal Year")
plt.xlabel("Fiscal Year")
plt.ylabel("Fraud Ratio")
plt.grid(True)
plt.show()
