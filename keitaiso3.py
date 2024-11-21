import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sudachipy import tokenizer
from sudachipy import dictionary
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import japanize_matplotlib
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

def analyze_frequent_words(texts, tokenizer_obj, mode, top_n=30):
    """
    頻出単語を分析する関数
    """
    word_counter = Counter()
    
    for text in texts:
        if pd.isna(text):
            continue
        tokens = tokenizer_obj.tokenize(str(text), mode)
        for token in tokens:
            pos = token.part_of_speech()[0]
            if pos in ['名詞', '動詞', '形容詞']:
                word_counter[token.dictionary_form()] += 1
    
    # 頻出単語とその出現回数を取得
    top_words = word_counter.most_common(top_n)
    
    # 可視化
    plt.figure(figsize=(15, 8))
    words, counts = zip(*top_words)
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('頻出単語トップ30')
    plt.xlabel('単語')
    plt.ylabel('出現回数')
    plt.tight_layout()
    plt.savefig('analysis_results/frequent_words.png')
    plt.close()
    
    return dict(top_words)

def find_optimal_clusters(tfidf_matrix, max_clusters=100, step=5):
    """
    エルボー法とシルエット分析でクラスター数を最適化する関数
    """
    inertias = []
    silhouette_scores = []
    n_clusters_range = range(5, max_clusters, step)
    
    for n_clusters in n_clusters_range:
        print(f"Testing {n_clusters} clusters...")
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            random_state=42
        )
        clusterer.fit(tfidf_matrix)
        inertias.append(clusterer.inertia_)
        
        # シルエットスコアの計算（サンプリングして計算時間を短縮）
        if tfidf_matrix.shape[0] > 10000:
            indices = np.random.choice(tfidf_matrix.shape[0], 10000, replace=False)
            score = silhouette_score(
                tfidf_matrix[indices], 
                clusterer.predict(tfidf_matrix[indices]),
                sample_size=5000
            )
        else:
            score = silhouette_score(
                tfidf_matrix, 
                clusterer.predict(tfidf_matrix),
                sample_size=5000
            )
        silhouette_scores.append(score)
    
    # エルボー法の可視化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(n_clusters_range, inertias, 'bx-')
    plt.xlabel('クラスター数')
    plt.ylabel('Inertia')
    plt.title('エルボー法')
    
    # シルエットスコアの可視化
    plt.subplot(1, 2, 2)
    plt.plot(n_clusters_range, silhouette_scores, 'rx-')
    plt.xlabel('クラスター数')
    plt.ylabel('シルエットスコア')
    plt.title('シルエット分析')
    
    plt.tight_layout()
    plt.savefig('analysis_results/cluster_optimization.png')
    plt.close()
    
    # 最適なクラスター数の選択（シルエットスコアが最大の値）
    optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    return optimal_clusters

def visualize_flag_analysis(flg_analysis, output_dir):
    """
    フラグデータの可視化を行う関数
    """
    for flg_col, analysis in flg_analysis.items():
        # 新規契約との関係
        plt.figure(figsize=(10, 6))
        analysis['new_contract_correlation'].plot(
            kind='bar',
            stacked=True,
            colormap='Set3'
        )
        plt.title(f'{flg_col}と新規契約の関係')
        plt.xlabel('新規契約')
        plt.ylabel('件数')
        plt.legend(title='フラグ値')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{flg_col}_new_contract_viz.png')
        plt.close()
        
        # 解約との関係
        plt.figure(figsize=(10, 6))
        analysis['cancellation_correlation'].plot(
            kind='bar',
            stacked=True,
            colormap='Set3'
        )
        plt.title(f'{flg_col}と解約の関係')
        plt.xlabel('解約')
        plt.ylabel('件数')
        plt.legend(title='フラグ値')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{flg_col}_cancellation_viz.png')
        plt.close()

def main():
    # Sudachiトークナイザーの初期化
    tokenizer_obj, mode = initialize_tokenizer()
    
    # データの読み込みと前処理
    print("Reading main data files...")
    main_df = read_csv_in_chunks('temp/*.csv')
    
    print("Reading auxiliary files...")
    hoge_df, huga_df, flg_columns = load_auxiliary_files()
    
    print("Merging dataframes...")
    # データの結合
    merged_df = main_df.merge(
        hoge_df,
        left_on='converted_contract_no',
        right_on='POL_NO',
        how='left'
    )
    
    merged_df = merged_df.merge(
        huga_df,
        left_on='converted_contract_no',
        right_on='POL_ID',
        how='left'
    )
    
    print("Analyzing frequent words...")
    # 頻出単語の分析
    frequent_words = analyze_frequent_words(merged_df['content1'], tokenizer_obj, mode)
    
    print("Processing text data...")
    # テキストの前処理
    merged_df['processed_content'] = merged_df['content1'].apply(
        lambda x: preprocess_text(x, tokenizer_obj, mode)
    )
    
    print("Vectorizing text data...")
    # TF-IDF変換
    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=3,
        max_df=0.7,
        ngram_range=(1, 3)
    )
    
    tfidf_matrix = vectorizer.fit_transform(merged_df['processed_content'])
    
    print("Finding optimal number of clusters...")
    # 最適なクラスター数の決定
    optimal_clusters = find_optimal_clusters(tfidf_matrix)
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    print("Performing clustering...")
    # クラスタリングの実行
    clusterer = MiniBatchKMeans(
        n_clusters=optimal_clusters,
        batch_size=1000,
        random_state=42
    )
    
    merged_df['content_cluster'] = clusterer.fit_predict(tfidf_matrix)
    
    # 新規契約と解約の判定
    print("Analyzing patterns...")
    merged_df['is_new_contract'] = merged_df['content1'].str.contains('ＮＢＳＲ', na=False)
    
    # クラスタごとの解約キーワード出現率を計算
    cancellation_keywords = ['解約', '解除', '退会', '失効', '停止', '中止']
    cluster_cancellation_rates = {}
    
    for cluster_id in merged_df['content_cluster'].unique():
        cluster_texts = merged_df[merged_df['content_cluster'] == cluster_id]['processed_content']
        keyword_freq = sum(
            cluster_texts.str.contains('|'.join(cancellation_keywords), 
                                    case=False, 
                                    na=False).sum()
        )
        cluster_cancellation_rates[cluster_id] = keyword_freq / len(cluster_texts)
    
    # 解約クラスタの特定
    cancellation_clusters = [
        cluster_id for cluster_id, rate in cluster_cancellation_rates.items() 
        if rate > 0.1
    ]
    
    merged_df['is_cancellation'] = merged_df['content_cluster'].isin(cancellation_clusters)
    
    # FLG列との関係分析
    print("Analyzing FLG relationships...")
    flg_analysis = {}
    for flg_col in flg_columns:
        flg_analysis[flg_col] = {
            'new_contract_correlation': pd.crosstab(
                merged_df['is_new_contract'],
                merged_df[flg_col]
            ),
            'cancellation_correlation': pd.crosstab(
                merged_df['is_cancellation'],
                merged_df[flg_col]
            )
        }
    
    # 結果の保存
    print("Saving results...")
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # フラグデータの可視化
    visualize_flag_analysis(flg_analysis, output_dir)
    
    # クラスタリング結果の特徴語を抽出して保存
    feature_names = vectorizer.get_feature_names_out()
    cluster_features = []
    
    for i in range(clusterer.n_clusters):
        center = clusterer.cluster_centers_[i]
        top_indices = center.argsort()[-5:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_features.append({
            'cluster_id': i,
            'top_terms': top_terms,
            'size': (merged_df['content_cluster'] == i).sum(),
            'cancellation_rate': cluster_cancellation_rates[i]
        })
    
    pd.DataFrame(cluster_features).to_csv(f'{output_dir}/cluster_features.csv', index=False)
    
    # 日次統計の保存
    daily_stats = merged_df.groupby(pd.to_datetime(merged_df['event_datetime']).dt.date).agg({
        'is_new_contract': 'sum',
        'is_cancellation': 'sum'
    })
    daily_stats.to_csv(f'{output_dir}/daily_stats.csv')
    
    # FLG分析結果の保存
    for flg_col, analysis in flg_analysis.items():
        analysis['new_contract_correlation'].to_csv(
            f'{output_dir}/{flg_col}_new_contract_analysis.csv'
        )
        analysis['cancellation_correlation'].to_csv(
            f'{output_dir}/{flg_col}_cancellation_analysis.csv'
        )
    
    print("Analysis complete!")
    return merged_df, cluster_features, flg_analysis, daily_stats, frequent_words

if __name__ == "__main__":
    merged_df, cluster_features, flg_analysis, daily_stats, frequent_words = main()
