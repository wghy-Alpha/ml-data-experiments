import pandas as pd
import os

# 1. 定义数据路径（请确保路径与你解压后的文件路径一致）
file_path = "ratings.csv"

if not os.path.exists(file_path):
    print("找不到 ratings.csv，请检查数据集是否已正确下载并解压。")
else:
    print("正在加载数据，这可能需要一点时间（约2500万条记录）...")
    # 读取评分数据
    # 使用 usecols 只读取需要的列，以节省内存和加快读取速度
    ratings = pd.read_csv(file_path, usecols=['movieId'])
    
    # 2. 计算基础统计信息
    total_ratings = len(ratings)
    total_movies = ratings['movieId'].nunique()
    
    print(f"系统总评分记录数: {total_ratings:,}")
    print(f"系统总电影（物品）数: {total_movies:,}")
    
    # 3. 统计每部电影的评分次数，并自动按降序排列
    movie_rating_counts = ratings['movieId'].value_counts()
    
    # 4. 计算 10% 的电影数量
    top_10_percent_movie_count = int(total_movies * 0.10)
    
    # 5. 获取这前 10% 热门电影的评分总数
    top_10_percent_ratings_sum = movie_rating_counts.head(top_10_percent_movie_count).sum()
    
    # 6. 计算占比
    percentage_of_total_ratings = (top_10_percent_ratings_sum / total_ratings) * 100
    
    print("-" * 30)
    print(f"前 10% 的热门电影数量: {top_10_percent_movie_count:,}")
    print(f"这 10% 的电影包含的评分数量: {top_10_percent_ratings_sum:,}")
    print(f"占比全体评分记录的比例: {percentage_of_total_ratings:.2f}%")
    
    # 验证结论
    if percentage_of_total_ratings > 75:
        print("\n结论验证【成功】：占比不足 10% 的热门电影确实包含了超过 75% 的有效评分！")
    else:
        print("\n结论验证【失败】：占比未达到 75%。")