import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import gc
import dask.dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from sudachipy import tokenizer
from sudachipy import dictionary
import warnings
warnings.filterwarnings('ignore')

def initialize_tokenizer():
    """
    Sudachi トークナイザーの初期化
    """
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C  # モードC（最も分割単位が大きい）を使用
    return tokenizer_obj, mode

def load_auxiliary_files():
    """
    補助ファイル（hoge.csv, huga.csv）を読み込む関数
    """
    # hogeファイルの読み込み（全ての列を保持）
    hoge_df = pd.read_csv('hoge.csv', dtype={'POL_NO': str})
    
    # FLGで終わる列を特定
    flg_columns = [col for col in hoge_df.columns if col.endswith('FLG')]
    print(f"Found FLG columns: {flg_columns}")
    
    # hugaファイルの読み込み
    huga_df = pd.read_csv('huga.csv', dtype={'POL_ID': str})
    
    return hoge_df, huga_df, flg_columns

def preprocess_text(text, tokenizer_obj, mode):
    """
    テキストの前処理を行う関数（Sudachiを使用）
    """
    if pd.isna(text):
        return ""
    
    # 基本的な前処理
    text = str(text).strip()
    
    # Sudachiによる分かち書き
    tokens = tokenizer_obj.tokenize(text, mode)
    
    # 形態素解析結果から必要な情報を抽出
    # 名詞、動詞、形容詞のbase形を取得
    words = []
    for token in tokens:
        pos = token.part_of_speech()[0]  # 品詞の大分類
        if pos in ['名詞', '動詞', '形容詞']:
            words.append(token.dictionary_form())
    
    return ' '.join(words)

def read_and_process_csv_chunks(file_pattern, chunksize=100000):
    """
    大きなCSVファイルを分割して読み込み、処理する関数
    """
    columns = [
        'event_datetime', 'seq_no', 'sales_person', 'data_datetime', 
        'content1', 'content2', 'company_type', 'contract_no', 
        'converted_contract_no', 'source_id', 'url1', 'url2', 
        'customer_id'] + [f'col_{i}' for i in range(14, 19)] + ['sales_person_id'] + [f'col_{i}' for i in range(20, 27)]
    
    ddf = dd.read_csv(file_pattern, 
                      names=columns,
                      encoding='utf-8',
                      dtype={
                          'converted_contract_no': str,
                          'company_type': str,
                          'content1': str,
                          'customer_id': str
                      })
    
    # 会社区分=Gのデータのみをフィルタリング
    ddf = ddf[ddf['company_type'] == 'G']
    
    # 必要な列を選択
    ddf = ddf[[
        'event_datetime', 'sales_person', 'content1', 
        'converted_contract_no', 'sales_person_id', 'customer_id'
    ]]
    
    return ddf

def cluster_content(df, tokenizer_obj, mode, n_clusters=50, random_state=42):
    """
    内容1の列をクラスタリングする関数
    """
    # テキストの前処理（Sudachiを使用）
    processed_texts = df['content1'].fillna('').apply(
        lambda x: preprocess_text(x, tokenizer_obj, mode)
    )
    
    # TF-IDF変換
    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=3,
        max_df=0.7,
        ngram_range=(1, 3)  # 1-3文字のn-gramを使用
    )
    
    # スパース行列に変換
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # MiniBatchKMeansでクラスタリング
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters, 
        batch_size=1000,
        random_state=random_state
    )
    
    df['content_cluster'] = clusterer.fit_predict(tfidf_matrix)
    
    # クラスタの特徴語を抽出
    cluster_features = []
    cluster_centers = clusterer.cluster_centers_
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(n_clusters):
        center = cluster_centers[i]
        top_indices = center.argsort()[-5:][::-1]  # 上位5つの特徴語
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_features.append({
            'cluster_id': i,
            'top_terms': top_terms,
            'size': (df['content_cluster'] == i).sum()
        })
    
    return df, cluster_features, tfidf_matrix, vectorizer

def identify_cancellation_clusters(df, cluster_features, tokenizer_obj, mode):
    """
    解約関連のクラスタを特定する関数
    """
    # 解約関連の一般的なキーワード
    cancellation_keywords = ['解約', '解除', '退会', '失効', '停止', '中止']
    
    # 各クラスタの解約関連度を計算
    cancellation_clusters = []
    for feature in cluster_features:
        cluster_id = feature['cluster_id']
        cluster_texts = df[df['content_cluster'] == cluster_id]['content1']
        
        # テキストの前処理とキーワードマッチング
        processed_texts = cluster_texts.apply(
            lambda x: preprocess_text(x, tokenizer_obj, mode)
        )
        
        # キーワードの出現頻度を計算
        keyword_freq = sum(
            processed_texts.str.contains('|'.join(cancellation_keywords), 
                                      case=False, 
                                      na=False).sum()
        )
        
        if keyword_freq / len(cluster_texts) > 0.1:  # 10%以上で解約関連と判定
            cancellation_clusters.append(cluster_id)
    
    return cancellation_clusters

def analyze_patterns(df, flg_columns, tokenizer_obj, mode):
    """
    新規契約、解約、およびFLGとの関係を分析する関数
    """
    # 新規契約の特定
    df['is_new_contract'] = df['content1'].str.contains('ＮＢＳＲ', na=False)
    
    # 解約クラスタに基づく解約フラグ
    df['is_cancellation'] = df['content_cluster'].isin(
        identify_cancellation_clusters(df, cluster_features, tokenizer_obj, mode)
    )
    
    # FLG列との関係分析
    flg_analysis = {}
    for flg_col in flg_columns:
        flg_analysis[flg_col] = {
            'new_contract_correlation': pd.crosstab(
                df['is_new_contract'],
                df[flg_col]
            ),
            'cancellation_correlation': pd.crosstab(
                df['is_cancellation'],
                df[flg_col]
            )
        }
    
    # 時系列分析
    daily_stats = df.groupby(pd.to_datetime(df['event_datetime']).dt.date).agg({
        'is_new_contract': 'sum',
        'is_cancellation': 'sum'
    })
    
    return flg_analysis, daily_stats

def main():
    # Sudachiトークナイザーの初期化
    tokenizer_obj, mode = initialize_tokenizer()
    
    # データの読み込みと前処理
    main_df = read_and_process_csv_chunks('temp/*.csv')
    hoge_df, huga_df, flg_columns = load_auxiliary_files()
    
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
    
    # テキストクラスタリングの実行
    merged_df, cluster_features, tfidf_matrix, vectorizer = cluster_content(
        merged_df, tokenizer_obj, mode
    )
    
    # パターン分析
    flg_analysis, daily_stats = analyze_patterns(
        merged_df, flg_columns, tokenizer_obj, mode
    )
    
    # 結果の保存
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # クラスタリング結果の保存
    cluster_df = pd.DataFrame(cluster_features)
    cluster_df.to_csv(f'{output_dir}/cluster_features.csv', index=False)
    
    # 日次統計の保存
    daily_stats.to_csv(f'{output_dir}/daily_stats.csv')
    
    # FLG分析結果の保存
    for flg_col, analysis in flg_analysis.items():
        analysis['new_contract_correlation'].to_csv(
            f'{output_dir}/{flg_col}_new_contract_analysis.csv'
        )
        analysis['cancellation_correlation'].to_csv(
            f'{output_dir}/{flg_col}_cancellation_analysis.csv'
        )
    
    # クラスタごとのサンプルデータ保存
    cluster_samples = merged_df.groupby('content_cluster').agg({
        'content1': lambda x: list(x.sample(min(5, len(x))))
    })
    cluster_samples.to_csv(f'{output_dir}/cluster_samples.csv')
    
    return merged_df, cluster_features, flg_analysis, daily_stats

if __name__ == "__main__":
    merged_df, cluster_features, flg_analysis, daily_stats = main()
