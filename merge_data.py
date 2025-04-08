import pandas as pd


'''
首先进行三方数据的处理，并合并。
管理层讨论与分析：是匹配的主要基础，保留bert结果、删除'经营分析时间'，只保留【年末】的分析结果代表当年特征。
财务数据：进行独热编码提取【行业】
欺诈数据：只作为label，单独提取年份并标注【是/否】发生欺诈

base_df最近的年份没有，而df_ar一直到2024年，因此会有一些无法匹配。
'''
# 管理层讨论分析数据处理
base_df = pd.read_csv('./data/管理层讨论与分析_bert.csv')

# 步骤1: 转换为日期格式并清理无效数据
base_df['经营分析时间'] = pd.to_datetime(
    base_df['经营分析时间'], 
    errors='coerce'  # 无效日期转为 NaT
)
base_df = base_df.dropna(subset=['经营分析时间'])

# 步骤2: 提取月份和日，筛选12-31的报告
is_year_end = (base_df['经营分析时间'].dt.month == 12)
df_year_end = base_df[is_year_end]

# 步骤3: 去重（如果某公司一年有多个12-31记录，保留第一条）
base_df = df_year_end.drop_duplicates(
    subset=['股票代码', '会计年度'], 
    keep='first'
)

base_df = base_df.drop(columns=['经营分析时间']) #不需要的数据

# df = df_ar
# print("=== 列名 ===")
# print(df.columns.tolist())
# print("\n=== 数据类型 ===")
# print(df.dtypes)
# print(df.head())


# 欺诈数据
df_ar = pd.read_csv('./data/AR_FINVIOLATION_cleaned.csv')

df_ar['违规年度'] = df_ar['违规年度'].str.strip(';')
df_ar['违规年度'] = df_ar['违规年度'].apply(
    lambda x: x.split(';') if isinstance(x, str) else []
)
df_ar['违规年度'] = df_ar['违规年度'].apply(
    lambda x: [year for year in x if year != 'N/A']
)
df_ar = df_ar[df_ar['违规年度'].apply(lambda x: len(x) > 0)]

df_ar = df_ar.explode('违规年度')

df_ar = df_ar[df_ar['违规年度'].str.strip() != ''] # 删除无效项

df_ar = df_ar.rename(columns={'违规年度': '会计年度'})
df_ar['是否违规'] = 1

df_ar = df_ar[['股票代码', '会计年度', '是否违规']] # 只保留label
df_ar = df_ar.drop_duplicates() # 去除重复

df_ar['会计年度'] = df_ar['会计年度'].astype(int)


# 会计数据
df_aiq = pd.read_csv('./data/AIQ_LCFinIndexY_cleaned.csv')

df_aiq.rename(columns={'证券代码': '股票代码'}, inplace=True)
df_aiq['会计年度'] = pd.to_datetime(df_aiq['统计截止日期']).dt.year
df_aiq = df_aiq.drop(columns=['统计截止日期'])


# 依次按股票代码和年份进行左连接

merged_df = pd.merge(base_df, df_ar, on=['股票代码', '会计年度'], how='left')
merged_df = pd.merge(merged_df, df_aiq, on=['股票代码', '会计年度'], how='left')


if '是否违规' in merged_df.columns:
    null_violation = merged_df['是否违规'].isna().sum()
    print(f"\n'是否违规'列空值数量: {null_violation}") # '是否违规'列空值数量: 48535（全部年末50783）


    # 填充空值为0
    merged_df['是否违规'] = merged_df['是否违规'].fillna(0)
    print("已填充空值为0")
else:
    print("\n警告：数据中不存在'是否违规'列")


# 输出
merged_df.to_csv('merged_data.csv', index=False)

print("合并完成，输出文件为merged_data.csv")

print(base_df.dtypes)
print(df_ar.dtypes)
print(df_aiq.dtypes)

