import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import random

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
plt.style.use('ggplot')

# 加载数据集（假设使用MovieLens 100k数据集）
ratings = pd.read_csv('ml-100k/u.data', sep='\t', 
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                    names=['item_id', 'title', 'release_date', 'video_release_date',
                          'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                          'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                          'Thriller', 'War', 'Western'])

# 合并评分和电影信息
data = pd.merge(ratings, movies, on='item_id')

# 计算每个物品的交互数
item_popularity = data['item_id'].value_counts().reset_index()
item_popularity.columns = ['item_id', 'interaction_count']

# 定义长尾和热门物品
tail_threshold = item_popularity['interaction_count'].quantile(0.1)
hot_threshold = item_popularity['interaction_count'].quantile(0.9)
long_tail_items = set(item_popularity[item_popularity['interaction_count'] <= tail_threshold]['item_id'])
hot_items = set(item_popularity[item_popularity['interaction_count'] >= hot_threshold]['item_id'])


user_hot_interest_similarities = []

genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

for user, group in data.groupby('user_id'):
    # 该用户交互过的所有物品兴趣向量（均值）
    all_interest = group[genre_cols].mean()
    # 该用户交互过的热门物品兴趣向量（均值）
    hot_group = group[group['item_id'].isin(hot_items)]
    if len(hot_group) == 0 or all_interest.sum() == 0:
        continue
    hot_interest = hot_group[genre_cols].mean()
    # 归一化
    if hot_interest.sum() > 0:
        all_interest = all_interest / all_interest.sum()
        hot_interest = hot_interest / hot_interest.sum()
        sim = cosine_similarity([all_interest.values], [hot_interest.values])[0][0]
        user_hot_interest_similarities.append(sim)

# 分析相似度
user_hot_interest_similarities = np.array(user_hot_interest_similarities)
print("\n[Hot items interest vs All items interest Cosine Similarity]")
print(f"Mean similarity: {user_hot_interest_similarities.mean():.3f}")
print(f"Median similarity: {np.median(user_hot_interest_similarities):.3f}")
print(f"Percentage with similarity > 0.5: {(user_hot_interest_similarities > 0.5).mean()*100:.1f}%")

plt.figure(figsize=(8, 4))
plt.hist(user_hot_interest_similarities, bins=20)
plt.title('Cosine Similarity: Hot Items Interest vs All Items Interest')
plt.xlabel('Cosine Similarity')
plt.ylabel('Number of Users')
plt.show()

# 随机选取10个用户
all_user_ids = list(data['user_id'].unique())
sample_users = random.sample(all_user_ids, 10)

print("\n===== 10个随机sample用户的热门物品兴趣和总体物品兴趣 =====")
for user in sample_users:
    group = data[data['user_id'] == user]
    all_interest = group[genre_cols].mean()
    hot_group = group[group['item_id'].isin(hot_items)]
    if len(hot_group) == 0 or all_interest.sum() == 0:
        print(f"User {user}: 无热门物品或兴趣向量全为0，跳过")
        continue
    hot_interest = hot_group[genre_cols].mean()
    # 归一化
    if hot_interest.sum() > 0:
        all_interest = all_interest / all_interest.sum()
        hot_interest = hot_interest / hot_interest.sum()
        print(f"\nUser {user}:")
        print("总体物品兴趣向量（前5维最大）:")
        print(all_interest.sort_values(ascending=False).head(5))
        print("热门物品兴趣向量（前5维最大）:")
        print(hot_interest.sort_values(ascending=False).head(5))