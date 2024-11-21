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

def plot_frequent_words(word_counter, top_n=30, fig_size=(15, 8)):
    """
    頻出単語をプロットする関数
    """
    plt.figure(figsize=fig_size)
    words, counts = zip(*word_counter.most_common(top_n))
    
    # 横向きの棒グラフを作成
    plt.barh(words, counts)
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    
    # レイアウトの調整
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
    
    # カバー率90%となる単語数を見つける
    coverage_threshold = 0.9
    optimal_count = np.where(cumsum / total_sum >= coverage_threshold)[0][0] + 1
    
    # カバー率のプロット
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
        
        # シルエットスコアの計算
        if features.shape[0] > 10000:
            # 大規模データセットの場合はサンプリング
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
    
    # エルボー法による最適クラスター数の検出
    kl = KneeLocator(
        list(n_clusters_range),
        inertias,
        curve='convex',
        direction='decreasing'
    )
    optimal_clusters_elbow = kl.elbow
    
    # 結果のプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # エルボーカーブ
    ax1.plot(n_clusters_range, inertias)
    ax1.axvline(x=optimal_clusters_elbow, color='r', linestyle='--', 
                label=f'Elbow point: {optimal_clusters_elbow}')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    ax1.legend()
    
    # シルエットスコア
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
    
    # エルボー法とシルエットスコアの結果を返す
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

def main():
    # 前のコードの初期化部分は同じ
    tokenizer_obj, mode = initialize_tokenizer()
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
    
    print("Extracting and analyzing frequent words...")
    # 単語頻度の分析
    word_counter = Counter()
    for text in merged_df['content1'].values:
        if pd.isna(text):
            continue
        tokens = tokenizer_obj.tokenize(str(text).strip(), mode)
        for token in tokens:
            pos = token.part_of_speech()[0]
            if pos in ['名詞', '動詞', '形容詞']:
                word_counter[token.dictionary_form()] += 1
    
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
        create_word_features(text, frequent_words)
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
    cluster_features = []
    for i in range(final_clusterer.n_clusters):
        cluster_docs = merged_df[merged_df['content_cluster'] == i]['content1']
        
        # クラスタ内の頻出単語を抽出
        cluster_words = Counter()
        for doc in cluster_docs:
            if pd.isna(doc):
                continue
            tokens = tokenizer_obj.tokenize(str(doc).strip(), mode)
            for token in tokens:
                pos = token.part_of_speech()[0]
                if pos in ['名詞', '動詞', '形容詞']:
                    cluster_words[token.dictionary_form()] += 1
        
        top_words = [word for word, _ in cluster_words.most_common(5)]
        
        cluster_features.append({
            'cluster_id': i,
            'top_terms': top_words,
            'size': len(cluster_docs),
            'sample_texts': cluster_docs.head(3).tolist()
        })
    
    # クラスターサイズの可視化
    cluster_sizes = visualize_cluster_sizes(merged_df, cluster_features)
    
    # 結果の保存
    print("Saving results...")
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 最適化結果の保存
    optimization_results = {
        'optimal_word_count': optimal_word_count,
        'optimal_clusters': optimal_clusters,
        'clustering_results': clustering_results
    }
    
    pd.DataFrame([optimization_results]).to_json(
        f'{output_dir}/optimization_results.json',
        orient='records'
    )
    
    # その他の結果の保存（前のコードと同様）
    pd.DataFrame(cluster_features).to_csv(f'{output_dir}/cluster_features.csv', index=False)
    pd.DataFrame({'frequent_words': frequent_words}).to_csv(
        f'{output_dir}/frequent_words.csv',
        index=False
    )
    
    cluster_stats = merged_df.groupby('content_cluster').agg({
        'content1': 'count',
        'sales_person': 'nunique',
        'customer_id': 'nunique'
    }).reset_index()
    
    cluster_stats.to_csv(f'{output_dir}/cluster_statistics.csv', index=False)
    
    print("Analysis complete!")
    return merged_df, cluster_features, frequent_words, cluster_stats, optimization_results

if __name__ == "__main__":
    merged_df, cluster_features, frequent_words, cluster_stats, optimization_results = main()
