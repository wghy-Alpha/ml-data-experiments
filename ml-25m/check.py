import os
import pandas as pd
print("文件存在:", os.path.exists('ratings.csv'))
print("文件大小:", os.path.getsize('ratings.csv'), "字节")

# 尝试读取前几行
try:
    sample = pd.read_csv('ratings.csv', encoding='latin-1', nrows=5)
    print("前5行数据:")
    print(sample.head())
except Exception as e:
    print("读取失败:", e)