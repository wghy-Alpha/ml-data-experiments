import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import random

# 加载评分数据
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')  # 假设有movies.csv，包含movieId和genres

# 提取评分大于等于4的数据
high_ratings = ratings[ratings['rating'] >= 4]

# 统计每个物品的交互数
item_interactions = high_ratings.groupby('movieId').size().reset_index(name='interaction_count')
item_interactions = item_interactions[item_interactions['interaction_count'] > 0]

# 预先构建 movieId 到 interaction_count 的映射字典
movieid2count = dict(zip(item_interactions['movieId'], item_interactions['interaction_count']))

# 处理类别为one-hot向量
all_genres = set()
for genres in movies['genres']:
    all_genres.update(genres.split('|'))
all_genres = sorted(all_genres)
genre2idx = {g: i for i, g in enumerate(all_genres)}

def genres_to_vector(genres):
    vec = [0] * len(all_genres)
    for g in genres.split('|'):
        if g in genre2idx:
            vec[genre2idx[g]] = 1
    return vec

movies['genre_vector'] = movies['genres'].apply(genres_to_vector)
movieid2vector = dict(zip(movies['movieId'], movies['genre_vector']))

# 统计每个用户的兴趣向量（用户级别的热门、长尾、随机）
user_interest = {}
for user, group in high_ratings.groupby('userId'):
    movie_ids = group['movieId'].tolist()
    if len(movie_ids) < 20:
        continue  # 交互物品太少跳过
    # 按全局交互数排序（用字典查找，效率高）
    movie_ids_sorted = sorted(
        movie_ids, 
        key=lambda x: movieid2count.get(x, 0), 
        reverse=True
    )
    n = len(movie_ids_sorted)
    top_n = max(1, int(n * 0.25))
    tail_n = max(1, int(n * 0.25))
    hot_cnt = 0
    tail_cnt = 0
    # 用户热门、长尾、随机物品
    user_hot = set(movie_ids_sorted[:top_n])
    user_longtail = set(movie_ids_sorted[-tail_n:])
    user_random = set(random.sample(movie_ids_sorted, top_n))
    # 向量初始化
    overall_vec = np.zeros(len(all_genres), dtype=int)
    hot_vec = np.zeros(len(all_genres), dtype=int)
    longtail_vec = np.zeros(len(all_genres), dtype=int)
    random_vec = np.zeros(len(all_genres), dtype=int)
    for mid in movie_ids:
        vec = movieid2vector.get(mid)
        if vec is None:
            continue
        overall_vec += np.array(vec)
        if mid in user_hot:
            hot_vec += np.array(vec)
            hot_cnt += 1
        if mid in user_longtail:
            longtail_vec += np.array(vec)
            tail_cnt += 1
        if mid in user_random:
            random_vec += np.array(vec)
    # 跳过没有交互过长尾或热门物品的用户，且长尾物品交互数需>=3
    if tail_cnt < 3 or hot_cnt < 3:
        continue
    user_interest[user] = {
        'overall': overall_vec.tolist(),
        'hot': hot_vec.tolist(),
        'longtail': longtail_vec.tolist(),
        'random': random_vec.tolist()
    }

# 对每个用户的四个兴趣向量做softmax
user_interest_softmax = {}
for user, vecs in user_interest.items():
    overall = softmax(vecs['overall'])
    hot = softmax(vecs['hot'])
    longtail = softmax(vecs['longtail'])
    randomv = softmax(vecs['random'])
    user_interest_softmax[user] = {
        'overall': overall,
        'hot': hot,
        'longtail': longtail,
        'random': randomv
    }

# 计算均值
cos_ol_list, cos_oh_list, cos_or_list = [], [], []
cos_hl_list, cos_hr_list = [], []
kl_ol_list, kl_oh_list, kl_or_list = [], [], []
kl_hl_list, kl_hr_list = [], []

epsilon = 1e-12
for user, vecs in user_interest_softmax.items():
    overall = vecs['overall']
    longtail = vecs['longtail']
    hot = vecs['hot']
    randomv = vecs['random']
    # 余弦相似度
    cos_ol_list.append(1 - cosine(overall, longtail))
    cos_oh_list.append(1 - cosine(overall, hot))
    cos_or_list.append(1 - cosine(overall, randomv))
    cos_hl_list.append(1 - cosine(hot, longtail))
    cos_hr_list.append(1 - cosine(hot, randomv))
    # KL散度
    kl_ol_list.append(entropy(overall + epsilon, longtail + epsilon))
    kl_oh_list.append(entropy(overall + epsilon, hot + epsilon))
    kl_or_list.append(entropy(overall + epsilon, randomv + epsilon))
    kl_hl_list.append(entropy(hot + epsilon, longtail + epsilon))
    kl_hr_list.append(entropy(hot + epsilon, randomv + epsilon))

print(f"总体-长尾 余弦相似度均值: {np.mean(cos_ol_list):.4f}  KL散度均值: {np.mean(kl_ol_list):.4f}")
print(f"总体-热门 余弦相似度均值: {np.mean(cos_oh_list):.4f}  KL散度均值: {np.mean(kl_oh_list):.4f}")
print(f"总体-随机 余弦相似度均值: {np.mean(cos_or_list):.4f}  KL散度均值: {np.mean(kl_or_list):.4f}")
print(f"热门-长尾 余弦相似度均值: {np.mean(cos_hl_list):.4f}  KL散度均值: {np.mean(kl_hl_list):.4f}")
print(f"热门-随机 余弦相似度均值: {np.mean(cos_hr_list):.4f}  KL散度均值: {np.mean(kl_hr_list):.4f}")




