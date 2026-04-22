import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import softmax
from scipy.stats import entropy
import csv
import tempfile
import os

def read_csv_skip_bad(path, expected_cols=None, encodings=('utf-8', 'cp1252', 'latin1')):
    """Robust CSV loader that tries multiple encodings and falls back to line-filtering to skip bad rows.

    - Tries pandas.read_csv with several encodings.
    - If ParserError occurs, tries engine='python' with on_bad_lines='skip' when available.
    - Otherwise creates a cleaned temporary CSV by reading with csv.reader (latin1 decode) and keeping only rows
      that have the expected number of columns (inferred from header if not provided).
    """
    from pandas.errors import ParserError

    # 1) quick attempts with pandas
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            # try next encoding
            continue
        except ParserError:
            # try python engine with on_bad_lines if available
            try:
                return pd.read_csv(path, encoding=enc, engine='python', on_bad_lines='skip')
            except TypeError:
                # older pandas may not support on_bad_lines
                break
            except Exception:
                break

    # 2) fallback: manual streaming + filter bad rows
    # infer expected columns from header if not provided
    if expected_cols is None:
        # open in text mode to properly read CSV header
        with open(path, 'r', encoding='latin1', errors='replace', newline='') as f:
            first = f.readline()
            expected_cols = len(list(csv.reader([first]))[0])

    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
    try:
        with open(path, 'r', encoding='latin1', errors='replace', newline='') as src, tmp:
            reader = csv.reader(src)
            writer = csv.writer(tmp)
            for row in reader:
                if len(row) == expected_cols:
                    writer.writerow(row)
        # read cleaned file with pandas (utf-8)
        df = pd.read_csv(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return df


# 加载评分数据（自动跳过坏行）
ratings = read_csv_skip_bad('ratings.csv', expected_cols=4)
movies = read_csv_skip_bad('movies.csv')  # 假设有movies.csv，包含movieId和genres

# 确保关键列为数值类型；若无法转换则置为 NaN，然后丢弃这些行
for col in ('userId', 'movieId', 'rating'):
    if col in ratings.columns:
        ratings[col] = pd.to_numeric(ratings[col], errors='coerce')
# 丢弃无法解析的行
ratings.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)
# 将 id 列转为整数（安全起见先转换为 int）
ratings['userId'] = ratings['userId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

# 提取评分大于等于4的数据
high_ratings = ratings[ratings['rating'] >= 4]

# 统计每个物品的交互数
item_interactions = high_ratings.groupby('movieId').size().reset_index(name='interaction_count')
item_interactions = item_interactions[item_interactions['interaction_count'] > 0]

# 热门物品：交互数排名前0.1%
top_01_percent = max(1, int(len(item_interactions) * 0.005))
hot_items = item_interactions.sort_values('interaction_count', ascending=False).head(top_01_percent)
hot_item_ids = set(hot_items['movieId'])

# 长尾物品：交互数排名后40%
tail_01_percent = max(1, int(len(item_interactions) * 0.4))
long_tail_items = item_interactions.sort_values('interaction_count', ascending=True).head(tail_01_percent)
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



# 对每个用户的三个兴趣向量（直接使用计数向量，不做softmax）
kl_ol_list, kl_oh_list, kl_hl_list = [], [], []
examples = []  # store up to 5 example users

# 对每个用户，裁剪掉三个向量都为0的维度，保持三向量长度一致；然后对裁剪后的向量做softmax并计算KL散度
epsilon = 1e-12
for user, vecs in user_interest.items():
    overall = np.array(vecs['overall'], dtype=float)
    longtail = np.array(vecs['longtail'], dtype=float)
    hot = np.array(vecs['hot'], dtype=float)

    # mask: 保留在任意一个向量中非零的维度
    mask = (overall != 0) | (longtail != 0) | (hot != 0)
    if not np.any(mask):
        # 三个向量全为零，跳过此用户
        continue
    overall_trim = overall[mask]
    longtail_trim = longtail[mask]
    hot_trim = hot[mask]

    # softmax 转换为概率分布（对每个向量独立softmax）
    overall_p = softmax(overall_trim)
    longtail_p = softmax(longtail_trim)
    hot_p = softmax(hot_trim)

    # 计算KL散度（使用 scipy.stats.entropy: KL(P||Q)）
    kl_ol = entropy(overall_p + epsilon, longtail_p + epsilon)
    kl_oh = entropy(overall_p + epsilon, hot_p + epsilon)
    kl_hl = entropy(hot_p + epsilon, longtail_p + epsilon)

    kl_ol_list.append(kl_ol)
    kl_oh_list.append(kl_oh)
    kl_hl_list.append(kl_hl)

    # collect up to 5 examples: store trimmed vectors and softmax results as lists
    if len(examples) < 5:
        examples.append({
            'user': user,
            'trimmed_overall': overall_trim.tolist(),
            'trimmed_longtail': longtail_trim.tolist(),
            'trimmed_hot': hot_trim.tolist(),
            'softmax_overall': overall_p.tolist(),
            'softmax_longtail': longtail_p.tolist(),
            'softmax_hot': hot_p.tolist(),
        })

# 对用户取平均并写入 1.txt
with open('1.txt', 'w', encoding='utf-8') as f:
    f.write('Examples (up to 5 users) after trimming and softmax:\n')
    for ex in examples:
        f.write(f"--- user: {ex['user']}\n")
        f.write('trimmed_overall: ' + str(ex['trimmed_overall']) + '\n')
        f.write('trimmed_longtail: ' + str(ex['trimmed_longtail']) + '\n')
        f.write('trimmed_hot: ' + str(ex['trimmed_hot']) + '\n')
        f.write('softmax_overall: ' + str(ex['softmax_overall']) + '\n')
        f.write('softmax_longtail: ' + str(ex['softmax_longtail']) + '\n')
        f.write('softmax_hot: ' + str(ex['softmax_hot']) + '\n')
        f.write('\n')
    f.write('Averaged KL divergences across users:\n')
    if len(kl_ol_list) == 0:
        f.write('kl_ol_mean: nan\n')
        f.write('kl_oh_mean: nan\n')
        f.write('kl_hl_mean: nan\n')
    else:
        f.write(f'kl_ol_mean: {np.mean(kl_ol_list):.6f}\n')
        f.write(f'kl_oh_mean: {np.mean(kl_oh_list):.6f}\n')
        f.write(f'kl_hl_mean: {np.mean(kl_hl_list):.6f}\n')




