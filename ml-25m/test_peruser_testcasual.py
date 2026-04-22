import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
import random

# 加载评分数据
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')  # 假设有movies.csv，包含movieId和genres
movies = movies[movies['genres'] != '(no genres listed)']

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
    # 随机挑选5%作为热门，5%作为长尾
    sample_num = max(1, int(n * 0.45))
    user_hot = set(random.sample(movie_ids_sorted, sample_num))
    user_longtail = set(random.sample([mid for mid in movie_ids_sorted if mid not in user_hot], sample_num))
    user_random = set(random.sample(movie_ids_sorted, sample_num))
    hot_cnt = 0
    tail_cnt = 0

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

# 对每个用户的四个兴趣向量（直接使用计数向量，不做softmax）
user_interest_proc = {}
for user, vecs in user_interest.items():
    overall = np.array(vecs['overall'], dtype=float)
    hot = np.array(vecs['hot'], dtype=float)
    longtail = np.array(vecs['longtail'], dtype=float)
    randomv = np.array(vecs['random'], dtype=float)
    user_interest_proc[user] = {
        'overall': overall,
        'hot': hot,
        'longtail': longtail,
        'random': randomv
    }

# 计算均值（只计算余弦相似度）
cos_hl_list = []
for user, vecs in user_interest_proc.items():
    hot = vecs['hot']
    longtail = vecs['longtail']
    # 余弦相似度
    cos_hl_list.append(1 - cosine(hot, longtail))

print(f"热门-长尾 余弦相似度均值: {np.mean(cos_hl_list):.4f}")






