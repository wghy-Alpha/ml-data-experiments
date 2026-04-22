import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

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

# 计算每个电影的评分次数
item_popularity = data['item_id'].value_counts().reset_index()
item_popularity.columns = ['item_id', 'interaction_count']

# 绘制物品流行度分布图（线性坐标）
plt.figure(figsize=(12, 6))
plt.hist(item_popularity['interaction_count'], bins=50)  # 去掉log=True
plt.title('Distribution of Item Popularity')
plt.xlabel('Number of Interactions')
plt.ylabel('Number of Items')
plt.show()

# 定义长尾物品：交互次数在后10%的物品
tail_threshold = item_popularity['interaction_count'].quantile(0.1)
long_tail_items = item_popularity[item_popularity['interaction_count'] <= tail_threshold]
print(f"Total items: {len(item_popularity)}")
print(f"Long-tail items (interactions <= {tail_threshold:.1f}): {len(long_tail_items)}")
print(f"Percentage of long-tail items: {len(long_tail_items)/len(item_popularity)*100:.1f}%")

# 为每个用户提取喜欢的电影类型（基于评分>=4的电影）
user_genre_pref = data[data['rating'] >= 4].groupby('user_id')[['Action', 'Adventure', 'Animation',
       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
       'Thriller', 'War', 'Western']].mean()

# 归一化处理，使每个用户的兴趣向量和为1
user_genre_pref = user_genre_pref.div(user_genre_pref.sum(axis=1), axis=0)

# 查看示例用户的兴趣特征
sample_user = user_genre_pref.sample(1)
print("\nSample user's genre preferences:")
print(sample_user.T.sort_values(by=sample_user.index[0], ascending=False).head(5))

# 获取用户交互过的长尾物品
user_long_tail_interactions = data[data['item_id'].isin(long_tail_items['item_id'])]

# 为每个用户提取其交互的长尾物品的特征
user_long_tail_features = user_long_tail_interactions.groupby('user_id')[['Action', 'Adventure', 'Animation',
       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
       'Thriller', 'War', 'Western']].mean()

# 归一化处理
user_long_tail_features = user_long_tail_features.div(user_long_tail_features.sum(axis=1), axis=0)

# 填充NaN值（对于没有长尾交互的用户）
user_long_tail_features = user_long_tail_features.reindex(user_genre_pref.index).fillna(0)

# 查看示例用户的长尾物品特征
print("\nSame user's long-tail item features:")
print(user_long_tail_features.loc[sample_user.index[0]].sort_values(ascending=False).head(5))

# 计算余弦相似度
similarities = []
for user in user_genre_pref.index:
    if user in user_long_tail_features.index and sum(user_long_tail_features.loc[user]) > 0:
        user_pref = user_genre_pref.loc[user].values.reshape(1, -1)
        user_long_tail = user_long_tail_features.loc[user].values.reshape(1, -1)
        similarity = cosine_similarity(user_pref, user_long_tail)[0][0]
        similarities.append(similarity)

# 分析相似度结果
similarities = np.array(similarities)
print("\nSimilarity analysis:")
print(f"Mean similarity: {similarities.mean():.3f}")
print(f"Median similarity: {np.median(similarities):.3f}")
print(f"Percentage with similarity > 0.5: {(similarities > 0.5).mean()*100:.1f}%")

# 绘制相似度分布
plt.figure(figsize=(10, 5))
plt.hist(similarities, bins=20)
plt.title('Distribution of Cosine Similarities between User Preferences and Long-tail Items')
plt.xlabel('Cosine Similarity')
plt.ylabel('Number of Users')
plt.show()

# 为每个用户采样与其交互过的长尾物品比例相同数量的随机物品，并计算相似度
random_similarities = []
for user in user_genre_pref.index:
    user_data = data[data['user_id'] == user]
    # 该用户交互过的长尾物品数量
    n_long_tail = user_long_tail_interactions[user_long_tail_interactions['user_id'] == user].shape[0]
    total = user_data.shape[0]
    # 计算该用户的长尾比例
    if total > 0:
        user_long_tail_ratio = n_long_tail / total
        n_sample = int(np.ceil(total * user_long_tail_ratio))
        # 至少采样1个
        if n_sample > 0:
            sampled = user_data.sample(n=n_sample, random_state=42)
            sampled_features = sampled[['Action', 'Adventure', 'Animation',
                                       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                       'Thriller', 'War', 'Western']].mean()
            # 归一化
            if sampled_features.sum() > 0:
                sampled_features = sampled_features / sampled_features.sum()
                user_pref = user_genre_pref.loc[user].values.reshape(1, -1)
                sampled_vec = sampled_features.values.reshape(1, -1)
                sim = cosine_similarity(user_pref, sampled_vec)[0][0]
                random_similarities.append(sim)

# 分析采样数据的相似度
random_similarities = np.array(random_similarities)
print("\nRandom sampling similarity analysis:")
print(f"Mean similarity: {random_similarities.mean():.3f}")
print(f"Median similarity: {np.median(random_similarities):.3f}")
print(f"Percentage with similarity > 0.5: {(random_similarities > 0.5).mean()*100:.1f}%")

# 可视化
plt.figure(figsize=(10, 5))
plt.hist(random_similarities, bins=20, alpha=0.7, label='Random Sample')
plt.hist(similarities, bins=20, alpha=0.7, label='Long-tail')
plt.title('Cosine Similarity: Random Sample vs Long-tail Items')
plt.xlabel('Cosine Similarity')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

# 定义热门物品：交互次数在前10%的物品
hot_threshold = item_popularity['interaction_count'].quantile(0.95)
hot_items = item_popularity[item_popularity['interaction_count'] >= hot_threshold]
print(f"Hot items (interactions >= {hot_threshold:.1f}): {len(hot_items)}")
print(f"Percentage of hot items: {len(hot_items)/len(item_popularity)*100:.1f}%")

# 获取用户交互过的热门物品
user_hot_interactions = data[data['item_id'].isin(hot_items['item_id'])]

# 为每个用户提取其交互的热门物品的特征
user_hot_features = user_hot_interactions.groupby('user_id')[['Action', 'Adventure', 'Animation',
       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
       'Thriller', 'War', 'Western']].mean()

# 归一化处理
user_hot_features = user_hot_features.div(user_hot_features.sum(axis=1), axis=0)
# 填充NaN值（对于没有热门交互的用户）
user_hot_features = user_hot_features.reindex(user_genre_pref.index).fillna(0)

# 查看示例用户的热门物品特征
print("\nSame user's hot item features:")
print(user_hot_features.loc[sample_user.index[0]].sort_values(ascending=False).head(5))

# 计算热门物品与用户兴趣的余弦相似度
hot_similarities = []
for user in user_genre_pref.index:
    if user in user_hot_features.index and user_hot_features.loc[user].sum() > 0:
        user_pref = user_genre_pref.loc[user].values.reshape(1, -1)
        user_hot = user_hot_features.loc[user].values.reshape(1, -1)
        similarity = cosine_similarity(user_pref, user_hot)[0][0]
        hot_similarities.append(similarity)

# 分析热门物品相似度结果
hot_similarities = np.array(hot_similarities)
print("\nHot items similarity analysis:")
print(f"Mean similarity: {hot_similarities.mean():.3f}")
print(f"Median similarity: {np.median(hot_similarities):.3f}")
print(f"Percentage with similarity > 0.5: {(hot_similarities > 0.5).mean()*100:.1f}%")

# 可视化
plt.figure(figsize=(10, 5))
plt.hist(hot_similarities, bins=20, alpha=0.7, label='Hot Items')
plt.hist(similarities, bins=20, alpha=0.7, label='Long-tail')
plt.hist(random_similarities, bins=20, alpha=0.7, label='Random Sample')
plt.title('Cosine Similarity: Hot Items vs Long-tail vs Random Sample')
plt.xlabel('Cosine Similarity')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

user_hot_similarities = []
user_tail_similarities = []

for user in user_genre_pref.index:
    user_data = data[data['user_id'] == user]
    # 合并全局流行度
    user_items_with_pop = user_data.merge(item_popularity, on='item_id')
    user_items_sorted = user_items_with_pop.sort_values('interaction_count', ascending=True).reset_index(drop=True)
    n_items = user_items_sorted.shape[0]
    if n_items < 2:
        continue
    n_tail = max(1, int(np.ceil(n_items * 0.1)))
    n_hot = max(1, int(np.ceil(n_items * 0.1)))
    # 后10%为长尾，前10%为热门
    user_long_tail_items = user_items_sorted.iloc[:n_tail]
    user_hot_items = user_items_sorted.iloc[-n_hot:]

    # 长尾特征
    tail_features = user_long_tail_items[['Action', 'Adventure', 'Animation',
                                          'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                          'Thriller', 'War', 'Western']].mean()
    if tail_features.sum() > 0:
        tail_features = tail_features / tail_features.sum()
        user_pref = user_genre_pref.loc[user].values.reshape(1, -1)
        tail_vec = tail_features.values.reshape(1, -1)
        sim_tail = cosine_similarity(user_pref, tail_vec)[0][0]
        user_tail_similarities.append(sim_tail)

    # 热门特征
    hot_features = user_hot_items[['Action', 'Adventure', 'Animation',
                                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                   'Thriller', 'War', 'Western']].mean()
    if hot_features.sum() > 0:
        hot_features = hot_features / hot_features.sum()
        user_pref = user_genre_pref.loc[user].values.reshape(1, -1)
        hot_vec = hot_features.values.reshape(1, -1)
        sim_hot = cosine_similarity(user_pref, hot_vec)[0][0]
        user_hot_similarities.append(sim_hot)

    # 计算热门与长尾向量的相似度
    if hot_features.sum() > 0 and tail_features.sum() > 0:
        hot_vec = hot_features.values.reshape(1, -1)
        tail_vec = tail_features.values.reshape(1, -1)
        hot_tail_sim = cosine_similarity(hot_vec, tail_vec)[0][0]
        # 可以选择保存到列表
        if 'hot_tail_similarities' not in locals():
            hot_tail_similarities = []
        hot_tail_similarities.append(hot_tail_sim)

# 分析
user_tail_similarities = np.array(user_tail_similarities)
user_hot_similarities = np.array(user_hot_similarities)
hot_tail_similarities = np.array(hot_tail_similarities)
print("\n[Per-user Tail Items Similarity]")
print(f"Mean similarity: {user_tail_similarities.mean():.3f}")
print(f"Median similarity: {np.median(user_tail_similarities):.3f}")
print(f"Percentage with similarity > 0.5: {(user_tail_similarities > 0.5).mean()*100:.1f}%")

print("\n[Per-user Hot Items Similarity]")
print(f"Mean similarity: {user_hot_similarities.mean():.3f}")
print(f"Median similarity: {np.median(user_hot_similarities):.3f}")
print(f"Percentage with similarity > 0.5: {(user_hot_similarities > 0.5).mean()*100:.1f}%")

print("\n[Hot vector vs Long-tail vector Similarity]")
print(f"Mean similarity: {hot_tail_similarities.mean():.3f}")
print(f"Median similarity: {np.median(hot_tail_similarities):.3f}")
print(f"Percentage with similarity > 0.5: {(hot_tail_similarities > 0.5).mean()*100:.1f}%")

# 可视化
plt.figure(figsize=(10, 5))
plt.hist(user_hot_similarities, bins=20, alpha=0.7, label='User Hot Items')
plt.hist(user_tail_similarities, bins=20, alpha=0.7, label='User Long-tail Items')
plt.title('Cosine Similarity: User Hot vs Long-tail Items')
plt.xlabel('Cosine Similarity')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

# 选取一个sample_user
sample_user_id = user_genre_pref.sample(1).index[0]
user_data = data[data['user_id'] == sample_user_id]
user_items_with_pop = user_data.merge(item_popularity, on='item_id')
user_items_sorted = user_items_with_pop.sort_values('interaction_count', ascending=True).reset_index(drop=True)
n_items = user_items_sorted.shape[0]
n_tail = max(1, int(np.ceil(n_items * 0.1)))
n_hot = max(1, int(np.ceil(n_items * 0.1)))
user_long_tail_items = user_items_sorted.iloc[:n_tail]
user_hot_items = user_items_sorted.iloc[-n_hot:]

# 计算向量
tail_vec = user_long_tail_items[['Action', 'Adventure', 'Animation',
                                 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                 'Thriller', 'War', 'Western']].mean()
hot_vec = user_hot_items[['Action', 'Adventure', 'Animation',
                          'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                          'Thriller', 'War', 'Western']].mean()

# 归一化
if tail_vec.sum() > 0:
    tail_vec = tail_vec / tail_vec.sum()
if hot_vec.sum() > 0:
    hot_vec = hot_vec / hot_vec.sum()

print(f"\nSample user id: {sample_user_id}")
print("Long-tail vector (new definition):")
print(tail_vec.sort_values(ascending=False).head(10))
print("\nHot vector (new definition):")
print(hot_vec.sort_values(ascending=False).head(10))

