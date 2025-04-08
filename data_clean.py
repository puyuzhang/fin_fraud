import pandas as pd


# 读取财务数据
file_path = "./data/AIQ_LCFinIndexY_raw.xlsx"
df = pd.read_excel(file_path, header=None, engine="openpyxl")

df.columns = df.iloc[1]

df = df.drop(index=[0, 1, 2]).reset_index(drop=True)

output_path = "./data/AIQ_LCFinIndexY_cleaned.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"处理完成，文件已保存为 {output_path}")


# 读取文本数据
file_path = "./data/管理层讨论与分析_raw.xlsx"
df = pd.read_excel(file_path, header=None, engine="openpyxl")

df.columns = df.iloc[1]

df = df.drop(index=[0, 1]).reset_index(drop=True)

output_path = "./data/管理层讨论与分析_cleaned.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"处理完成，文件已保存为 {output_path}")


# 读取欺诈数据
file_path = "./data/AR_FINVIOLATION_raw.xlsx"
df = pd.read_excel(file_path, header=None, engine="openpyxl")

df.columns = df.iloc[1]

df = df.drop(index=[0, 1, 2]).reset_index(drop=True)

output_path = "./data/AR_FINVIOLATION_cleaned.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"处理完成，文件已保存为 {output_path}")



# 数据检查
file_path = "./data/AR_FINVIOLATION_cleaned.csv"
df = pd.read_csv(file_path, dtype=str)

print(df.head(50))

print(df.dtypes)