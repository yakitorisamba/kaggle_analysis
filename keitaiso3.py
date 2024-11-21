import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sudachipy import tokenizer
from sudachipy import dictionary
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import japanize_matplotlib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def initialize_tokenizer():
    """
    Sudachi トークナイザーの初期化
    """
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    return tokenizer_obj, mode

def preprocess_text(text, tokenizer_obj, mode):
    """
    テキストの前処理を行う関数
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    tokens = tokenizer_obj.tokenize(text, mode)
    words = []
    for token in tokens:
        pos = token.part_of_speech()[0]
        if pos in ['名詞', '動詞', '形容詞']:
            words.append(token.dictionary_form())
    
    return ' '.join(words)

def extract_frequent_words(texts, tokenizer_obj, mode, top_n=1000):
    """
    テキストから頻出単語を抽出する関数
    """
    word_counter = Counter()
    
    for text in tqdm(texts, desc="Extracting frequent words"):
        if pd.isna(text):
            continue
        tokens = tokenizer_obj.tokenize(str(text).strip(), mode)
        for token in tokens:
            pos = token.part_of_speech()[0]
            if pos in ['名詞', '動詞', '形容詞']:
                word_counter[token.dictionary_form()] += 1
    
    return word_counter

def create_word_features(text, frequent_words):
    """
    テキストから頻出単語の特徴量を作成する関数
    """
    features = {word: 0 for word in frequent_words}
    if pd.isna(text):
        return features
    
    text = str(text).strip()
    for word in text.split():
        if word in features:
            features[word] = 1
    
    return features

def read_csv_in_chunks(file_pattern, chunk_size=100000):
    """
    CSVファイルを分割して読み込む関数
    """
    all_chunks = []
    columns = [
        'event_datetime', 'seq_no', 'sales_person', 'data_datetime', 
        'content1', 'content2', 'company_type', 'contract_no', 
        'converted_contract_no', 'source_id', 'url1', 'url2', 
        'customer_id'] + [f'col_{i}' for i in range(14, 19)] + ['sales_person_id'] + [f'col_{i}' for i in range(20, 27)]
    
    for file in glob.glob(file_pattern):
        print(f"Processing file: {file}")
        chunks = pd.read_csv(file, 
                           names=columns,
                           chunksize=chunk_size,
                           dtype={
                               'converted_contract_no': str,
                               'company_type': str,
                               'content1': str,
                               'customer_id': str
                           })
        
        for chunk in chunks:
            filtered_chunk = chunk[chunk['company_type'] == 'G']
            needed_columns = [
                'event_datetime', 'sales_person', 'content1', 
                'converted_contract_no', 'sales_person_id', 'customer_id'
            ]
            all_chunks.append(filtered_chunk[needed_columns])
            
        print(f"Current memory usage: {sum([chunk.memory_usage().sum() for chunk in all_chunks]) / 1024**2:.2f} MB")
    
    return pd.concat(all_chunks, ignore_index=True)

def load_auxiliary_files():
    """
    補助ファイルを読み込む関数
    """
    hoge_df = pd.read_csv('hoge.csv', dtype={'POL_NO': str})
    flg_columns = [col for col in hoge_df.columns if col.endswith('FLG')]
    print(f"Found FLG columns: {flg_columns}")
    
    huga_df = pd.read_csv('huga.csv', dtype={'POL_ID': str})
    
    return hoge_df, huga_df, flg_columns

def plot_frequent_words(word_counter, top_n=30, fig_size=(15, 8)):
    """
    頻出単語をプロットする関数
    """
    plt.figure(figsize=fig_size)
    words, counts = zip(*word_counter.most_common(top_n))
    
    plt.barh(words, counts)
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    
    plt.tight_layout()
    plt.savefig('analysis_results/frequent_words_plot.png')
    plt.close()

def plot_word_distribution(word_counter, fig_size=(12, 6)):
    """
    単語の出現頻度分布をプロットする関数
    """
    frequencies = np.array([count for _, count in word_counter.most_common()])
    
    plt.figure(figsize=fig_size)
    plt.plot(range(1, len(frequencies) + 1), frequencies)
    plt.title('Word Frequency Distribution')
    plt.xlabel('Word Rank')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analysis_results/word_distribution_plot.png')
    plt.close()

def find_optimal_word_count(word_counter, max_words=2000):
    """
    最適な頻出単語数を見つける関数
    """
    frequencies = [count for _, count in word_counter.most_common(max_words)]
    cumsum = np.cumsum(frequencies)
    total_sum = cumsum[-1]
    
    coverage_threshold = 0.9
    optimal_count = np.where(cumsum / total_sum >= coverage_threshold)[0][0] + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum) + 1), cumsum / total_sum)
    plt.axhline(y=coverage_threshold, color='r', linestyle='--', 
                label=f'{coverage_threshold*100}% Coverage')
    plt.axvline(x=optimal_count, color='g', linestyle='--', 
                label=f'Optimal count: {optimal_count}')
    plt.title('Word Coverage Analysis')
    plt.xlabel('Number of Words')
    plt.ylabel('Cumulative Coverage')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis_results/word_coverage_plot.png')
    plt.close()
    
    return optimal_count

def find_optimal_clusters(features, max_clusters=100, step=5):
    """
    エルボー法とシルエット分析でクラスター数を最適化する関数
    """
    inertias = []
    silhouette_scores = []
    n_clusters_range = range(2, max_clusters, step)
    
    for n_clusters in tqdm(n_clusters_range, desc="Finding optimal clusters"):
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            random_state=42
        )
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        
        if features.shape[0] > 10000:
            sample_size = 10000
            indices = np.random.choice(features.shape[0], sample_size, replace=False)
            score = silhouette_score(
                features[indices], 
                kmeans.predict(features[indices]),
                sample_size=sample_size
            )
        else:
            score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)
    
    kl = KneeLocator(
        list(n_clusters_range),
        inertias,
        curve='convex',
        direction='decreasing'
    )
    optimal_clusters_elbow = kl.elbow
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(n_clusters_range, inertias)
    ax1.axvline(x=optimal_clusters_elbow, color='r', linestyle='--', 
                label=f'Elbow point: {optimal_clusters_elbow}')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(n_clusters_range, silhouette_scores)
    best_silhouette = n_clusters_range[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_silhouette, color='r', linestyle='--', 
                label=f'Best score: {best_silhouette}')
    ax2.set_title('Silhouette Analysis')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('analysis_results/cluster_optimization_plot.png')
    plt.close()
    
    return {
        'elbow_clusters': optimal_clusters_elbow,
        'silhouette_clusters': best_silhouette,
        'silhouette_scores': dict(zip(n_clusters_range, silhouette_scores))
    }

def visualize_cluster_sizes(merged_df, cluster_features, fig_size=(12, 6)):
    """
    クラスターサイズの分布を可視化する関数
    """
    cluster_sizes = merged_df['content_cluster'].value_counts().sort_index()
    
    plt.figure(figsize=fig_size)
    plt.bar(range(len(cluster_sizes)), cluster_sizes.values)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Documents')
    plt.tight_layout()
    plt.savefig('analysis_results/cluster_sizes_plot.png')
    plt.close()
    
    return cluster_sizes

def analyze_cluster_features(merged_df, cluster_id, tokenizer_obj, mode):
    """
    特定のクラスタの特徴を分析する関数
    """
    cluster_docs = merged_df[merged_df['content_cluster'] == cluster_id]['content1']
    
    # クラスタ内の頻出単語を抽出
    word_counter = Counter()
    for doc in cluster_docs:
        if pd.isna(doc):
            continue
        tokens = tokenizer_obj.tokenize(str(doc).strip(), mode)
        for token in tokens:
            pos = token.part_of_speech()[0]
            if pos in ['名詞', '動詞', '形容詞']:
                word_counter[token.dictionary_form()] += 1
    
    return {
        'size': len(cluster_docs),
        'top_words': [word for word, _ in word_counter.most_common(5)],
        'word_frequencies': dict(word_counter.most_common(20)),
        'sample_texts': cluster_docs.head(3).tolist()
    }

def main():
    # Sudachiトークナイザーの初期化
    tokenizer_obj, mode = initialize_tokenizer()
    
    # データの読み込みと前処理
    print("Reading main data files...")
    main_df = read_csv_in_chunks('temp/*.csv')
    
    print("Reading auxiliary files...")
    hoge_df, huga_df, flg_columns = load_auxiliary_files()
    
    print("Merging dataframes...")
    merged_df = main_df.merge(
        hoge_df,
        left_on='converted_contract_no',
        right_on='POL_NO',
        how='left'
    ).merge(
        huga_df,
        left_on='converted_contract_no',
        right_on='POL_ID',
        how='left'
    )
    
    print("Analyzing word frequencies...")
    # 単語頻度の分析
    word_counter = extract_frequent_words(merged_df['content1'].values, tokenizer_obj, mode)
    
    # 頻出単語の可視化
    plot_frequent_words(word_counter)
    plot_word_distribution(word_counter)
    
    # 最適な単語数の決定
    optimal_word_count = find_optimal_word_count(word_counter)
    print(f"Optimal number of frequent words: {optimal_word_count}")
    
    # 最適化された単語リストの取得
    frequent_words = [word for word, _ in word_counter.most_common(optimal_word_count)]
    
    print("Creating word features...")
    # 特徴量の作成
    word_features = pd.DataFrame([
        create_word_features(
            preprocess_text(text, tokenizer_obj, mode), 
            frequent_words
        )
        for text in tqdm(merged_df['content1'].values, desc="Processing texts")
    ])
    
    # 特徴量を numpy array に変換
    features_array = word_features.values
    
    print("Finding optimal number of clusters...")
    # 最適なクラスター数の探索
    clustering_results = find_optimal_clusters(features_array)
    
    # エルボー法とシルエットスコアの結果を比較して最終的なクラスター数を決定
    optimal_clusters = clustering_results['elbow_clusters']
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    print("Performing final clustering...")
    # 最適化されたパラメータでクラスタリングを実行
    final_clusterer = MiniBatchKMeans(
        n_clusters=optimal_clusters,
        batch_size=1000,
        random_state=42
    )
    
    merged_df['content_cluster'] = final_clusterer.fit_predict(features_array)
    
    # クラスタの特徴分析
    print("Analyzing cluster features...")
    cluster_features = []
    for i in range(final_clusterer.n_clusters):
        features = analyze_cluster_features(merged_df, i, tokenizer_
