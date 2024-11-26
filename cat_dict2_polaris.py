import polars as pl
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
    return pl.read_csv(filepath, encoding='utf-8')

def classify_content(text, category_dict):
    """
    テキストを分類辞書に基づいて分類する関数
    
    Args:
        text (str): 分類対象のテキスト
        category_dict (DataFrame): 分類辞書
        
    Returns:
        tuple: (小分類, 中分類)
    """
    if text is None:
        return None, None
        
    text = str(text)
    # Polarisでのイテレーション方法に変更
    for row in category_dict.iter_rows():
        if row[0] in text:
            return row[0], row[1]
            
    return None, None

def check_complaint(text):
    """
    苦情の有無をチェックする関数
    """
    if text is None:
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
        # Polarisでの読み込み方法に変更
        df = pl.read_csv(file, 
                        has_header=False,
                        new_columns=columns,
                        dtypes={
                            'converted_contract_no': pl.Utf8,
                            'company_type': pl.Utf8,
                            'content1': pl.Utf8,
                            'content2': pl.Utf8,
                            'customer_id': pl.Utf8
                        })
        
        # 会社タイプGのフィルタリング
        filtered_df = df.filter(pl.col('company_type') == 'G')
        needed_columns = [
            'event_datetime', 'sales_person', 'content1', 'content2',
            'converted_contract_no', 'sales_person_id', 'customer_id'
        ]
        all_chunks.append(filtered_df.select(needed_columns))
    
    return pl.concat(all_chunks)

def main():
    print("Loading category dictionary...")
    category_dict = load_category_dictionary('category_dict.csv')
    
    print("Reading main data files...")
    main_df = read_csv_in_chunks('temp/*.csv')
    
    print("Classifying contents...")
    # content1の分類
    # Polarisでの処理方法に変更
    classifications = [classify_content(text, category_dict) 
                      for text in main_df.get_column('content1')]
    small_categories, medium_categories = zip(*classifications)
    
    main_df = main_df.with_columns([
        pl.Series('small_category', small_categories),
        pl.Series('medium_category', medium_categories)
    ])
    
    # content2の苦情チェック
    main_df = main_df.with_columns([
        pl.col('content2').map_elements(check_complaint).alias('is_complaint')
    ])
    
    # 分類結果の集計
    print("Aggregating results...")
    category_stats = (
        main_df.groupby(['medium_category', 'small_category'])
        .agg([
            pl.col('content1').count().alias('content1'),
            pl.col('is_complaint').sum().alias('is_complaint'),
            pl.col('sales_person').n_unique().alias('sales_person'),
            pl.col('customer_id').n_unique().alias('customer_id')
        ])
    )
    
    # 結果の保存
    print("Saving results...")
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    category_stats.write_csv(f'{output_dir}/category_statistics.csv')
    
    # 分類結果の可視化
    plt.figure(figsize=(15, 8))
    # Polarisでの集計方法に変更
    category_counts = (
        category_stats.groupby('medium_category')
        .agg(pl.col('content1').sum())
    )
    plt.bar(range(len(category_counts)), category_counts['content1'].to_list())
    plt.xticks(range(len(category_counts)), category_counts['medium_category'].to_list(), rotation=45)
    plt.title('Distribution of Medium Categories')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_distribution.png')
    plt.close()
    
    print("Analysis complete!")
    return main_df, category_stats

if __name__ == "__main__":
    main_df, category_stats = main()
