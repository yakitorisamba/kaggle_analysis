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
            # 会社区分=Gのデータのみをフィルタリング
            filtered_chunk = chunk[chunk['company_type'] == 'G']
            needed_columns = [
                'event_datetime', 'sales_person', 'content1', 
                'converted_contract_no', 'sales_person_id', 'customer_id'
            ]
            all_chunks.append(filtered_chunk[needed_columns])
            
        # メモリ使用量を表示
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
    
    print("Processing text data...")
    # テキストの前処理
    merged_df['processed_content'] = merged_df['content1'].apply(
        lambda x: preprocess_text(x, tokenizer_obj, mode)
    )
    
    print("Performing clustering...")
    # TF-IDF変換とクラスタリング
    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=3,
        max_df=0.7,
        ngram_range=(1, 3)
    )
    
    tfidf_matrix = vectorizer.fit_transform(merged_df['processed_content'])
    
    clusterer = MiniBatchKMeans(
        n_clusters=50,
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
    return merged_df, cluster_features, flg_analysis, daily_stats

if __name__ == "__main__":
    merged_df, cluster_features, flg_analysis, daily_stats = main()
