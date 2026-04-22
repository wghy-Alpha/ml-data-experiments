import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cosine
from scipy.stats import entropy

# 加载评分数据
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')  # 假设有movies.csv，包含movieId和genres

# 提取评分大于等于4的数据
high_ratings = ratings[ratings['rating'] >= 4]

# 统计每个物品的交互数
item_interactions = high_ratings.groupby('movieId').size().reset_index(name='interaction_count')
item_interactions = item_interactions[item_interactions['interaction_count'] > 0]

# 热门物品：交互数排名前0.1%
top_01_percent = max(1, int(len(item_interactions) * 0.001))
hot_items = item_interactions.sort_values('interaction_count', ascending=False).head(top_01_percent)
hot_item_ids = set(hot_items['movieId'])

# 长尾物品：交互数排名后40%
tail_40_percent = max(1, int(len(item_interactions) * 0.4))
long_tail_items = item_interactions.sort_values('interaction_count', ascending=True).head(tail_40_percent)
long_tail_item_ids = set(long_tail_items['movieId'])

# 输出热门物品和长尾物品数量
print(f"热门物品数量: {len(hot_items)}")
print(f"长尾物品数量: {len(long_tail_items)}")

# 输出热门物品的交互数降序分布
print("热门物品交互数降序分布：")
print(item_interactions[item_interactions['movieId'].isin(hot_item_ids)].sort_values('interaction_count', ascending=False))

# 输出长尾物品的交互数降序分布
print("长尾物品交互数降序分布：")
print(item_interactions[item_interactions['movieId'].isin(long_tail_item_ids)].sort_values('interaction_count', ascending=False))



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

# 统计每个用户的兴趣向量
user_interest = {}
for user, group in high_ratings.groupby('userId'):
    movie_ids = group['movieId'].tolist()
    # 总体兴趣
    overall_vec = [0] * len(all_genres)
    # 长尾兴趣
    longtail_vec = [0] * len(all_genres)
    # 热门兴趣
    hot_vec = [0] * len(all_genres)
    longtail_count = 0  # 统计长尾物品交互数
    for mid in movie_ids:
        vec = movieid2vector.get(mid)
        if vec is None:
            continue
        overall_vec = [a + b for a, b in zip(overall_vec, vec)]
        if mid in long_tail_item_ids:
            longtail_vec = [a + b for a, b in zip(longtail_vec, vec)]
            longtail_count += 1
        if mid in hot_item_ids:
            hot_vec = [a + b for a, b in zip(hot_vec, vec)]
    # 跳过没有交互过长尾或热门物品的用户，且长尾物品交互数需>=3
    if longtail_count < 3 or sum(hot_vec) == 0:
        continue
    user_interest[user] = {
        'overall': overall_vec,
        'longtail': longtail_vec,
        'hot': hot_vec
    }



# 对每个用户的三个兴趣向量做softmax
user_interest_softmax = {}
for user, vecs in user_interest.items():
    overall = softmax(vecs['overall'])
    longtail = softmax(vecs['longtail'])
    hot = softmax(vecs['hot'])
    user_interest_softmax[user] = {
        'overall': overall,
        'longtail': longtail,
        'hot': hot
    }

cos_ol_list, cos_oh_list, cos_hl_list = [], [], []
kl_ol_list, kl_oh_list, kl_hl_list = [], [], []

epsilon = 1e-12
# 计算两两之间的余弦相似度和KL散度
for user, vecs in user_interest_softmax.items():
    overall = vecs['overall']
    longtail = vecs['longtail']
    hot = vecs['hot']
    # 余弦相似度
    sim_ol = 1 - cosine(overall, longtail)
    sim_oh = 1 - cosine(overall, hot)
    sim_hl = 1 - cosine(hot, longtail)
    cos_ol_list.append(sim_ol)
    cos_oh_list.append(sim_oh)
    cos_hl_list.append(sim_hl)
    # KL散度（加epsilon防止inf）
    kl_ol = entropy(overall + epsilon, longtail + epsilon)
    kl_oh = entropy(overall + epsilon, hot + epsilon)
    kl_hl = entropy(hot + epsilon, longtail + epsilon)
    kl_ol_list.append(kl_ol)
    kl_oh_list.append(kl_oh)
    kl_hl_list.append(kl_hl)

print(f"总体-长尾 余弦相似度均值: {np.mean(cos_ol_list):.4f}  KL散度均值: {np.mean(kl_ol_list):.4f}")
print(f"总体-热门 余弦相似度均值: {np.mean(cos_oh_list):.4f}  KL散度均值: {np.mean(kl_oh_list):.4f}")
print(f"长尾-热门 余弦相似度均值: {np.mean(cos_hl_list):.4f}  KL散度均值: {np.mean(kl_hl_list):.4f}")




