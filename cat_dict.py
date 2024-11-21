import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_category_dictionary(filepath):
    """
    カテゴリー分類辞書を読み込む関数
    Returns:
        DataFrame: 分類辞書（1列目: 検索文字, 2列目: 中分類）
    """
    return pd.read_csv(filepath, encoding='utf-8')

def classify_content(text, category_dict):
    """
    テキストを分類辞書に基づいて分類する関数
    
    Args:
        text (str): 分類対象のテキスト
        category_dict (DataFrame): 分類辞書
        
    Returns:
        tuple: (小分類, 中分類)
    """
    if pd.isna(text):
        return None, None
        
    text = str(text)
    for _, row in category_dict.iterrows():
        if row.iloc[0] in text:
            return row.iloc[0], row.iloc[1]
            
    return None, None

def check_complaint(text):
    """
    苦情の有無をチェックする関数
    """
    if pd.isna(text):
        return False
    return '苦情' in str(text)

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
                               'content2': str,
                               'customer_id': str
                           })
        
        for chunk in chunks:
            filtered_chunk = chunk[chunk['company_type'] == 'G']
            needed_columns = [
                'event_datetime', 'sales_person', 'content1', 'content2',
                'converted_contract_no', 'sales_person_id', 'customer_id'
            ]
            all_chunks.append(filtered_chunk[needed_columns])
    
    return pd.concat(all_chunks, ignore_index=True)

def main():
    print("Loading category dictionary...")
    category_dict = load_category_dictionary('category_dict.csv')
    
    print("Reading main data files...")
    main_df = read_csv_in_chunks('temp/*.csv')
    
    print("Classifying contents...")
    # content1の分類
    classifications = [classify_content(text, category_dict) 
                      for text in main_df['content1']]
    main_df['small_category'] = [c[0] for c in classifications]
    main_df['medium_category'] = [c[1] for c in classifications]
    
    # content2の苦情チェック
    main_df['is_complaint'] = main_df['content2'].apply(check_complaint)
    
    # 分類結果の集計
    print("Aggregating results...")
    category_stats = main_df.groupby(['medium_category', 'small_category']).agg({
        'content1': 'count',
        'is_complaint': 'sum',
        'sales_person': 'nunique',
        'customer_id': 'nunique'
    }).reset_index()
    
    # 結果の保存
    print("Saving results...")
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    category_stats.to_csv(f'{output_dir}/category_statistics.csv', index=False)
    
    # 分類結果の可視化
    plt.figure(figsize=(15, 8))
    category_counts = category_stats.groupby('medium_category')['content1'].sum()
    plt.bar(range(len(category_counts)), category_counts.values)
    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45)
    plt.title('Distribution of Medium Categories')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_distribution.png')
    plt.close()
    
    print("Analysis complete!")
    return main_df, category_stats

if __name__ == "__main__":
    main_df, category_stats = main()
