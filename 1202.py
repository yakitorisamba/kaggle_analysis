import polars as pl
import glob
import os
from datetime import datetime

def read_csv_in_chunks(file_pattern):
    """
    CSVファイルを読み込み、重複行を適切に処理する関数
    
    Args:
        file_pattern (str): 読み込むCSVファイルのパターン
        
    Returns:
        pl.DataFrame: 処理済みのDataFrame
    """
    columns = [
        'event_datetime', 'seq_no', 'sales_person', 'data_datetime', 
        'content1', 'content2', 'company_type', 'contract_no', 
        'converted_contract_no', 'source_id', 'url1', 'url2', 
        'customer_id'] + [f'col_{i}' for i in range(14, 19)] + ['sales_person_id'] + [f'col_{i}' for i in range(20, 27)]
    
    all_files = glob.glob(file_pattern)
    if not all_files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    all_dfs = []
    
    for file in all_files:
        print(f"Processing file: {file}")
        df = pl.read_csv(
            file, 
            has_header=False, 
            new_columns=columns, 
            encoding='cp932'
        ).filter(pl.col('company_type') == 'G').select([
            'event_datetime', 'sales_person', 'content1', 'content2',
            'converted_contract_no', 'sales_person_id', 'customer_id'
        ])
        
        # 同一ファイル内での重複行の処理
        key_columns = ['event_datetime', 'converted_contract_no']
        
        for col in df.columns:
            if col not in key_columns:
                # 重複する行のグループごとに異なる値を持つものを特定
                duplicate_groups = (df
                    .group_by(key_columns)
                    .agg([
                        pl.col(col).alias('values'),
                        pl.count().alias('count')
                    ])
                    .filter(pl.col('count') > 1)
                )
                
                if len(duplicate_groups) > 0:
                    # 各グループ内で実際に値が異なるものだけを処理
                    for group in duplicate_groups.iter_rows():
                        values = group['values']
                        unique_values = list(set(values))
                        
                        if len(unique_values) > 1:
                            # 異なる値がある場合のみ、インデックス付き列を作成
                            for idx, value in enumerate(unique_values, 1):
                                df = df.with_columns([
                                    pl.when(
                                        (pl.col('event_datetime') == group[0]) &
                                        (pl.col('converted_contract_no') == group[1]) &
                                        (pl.col(col) == value)
                                    )
                                    .then(value)
                                    .otherwise(None)
                                    .alias(f'{col}_{idx}')
                                ])
        
        # 重複行を削除（キー列のみを基準とする）
        df = df.unique(subset=key_columns)
        all_dfs.append(df)
    
    # すべてのファイルのDataFrameを結合
    combined_df = pl.concat(all_dfs)
    return combined_df

def load_auxiliary_files():
    """
    補助ファイルを読み込み、重複行を適切に処理する関数
    
    Returns:
        tuple: (hoge_df, huga_df, flg_columns)
    """
    hoge_df = pl.read_csv('hoge.csv', encoding='cp932')
    flg_columns = [col for col in hoge_df.columns if col.endswith('FLG')]
    print(f"Found FLG columns: {flg_columns}")
    
    # hugaファイルの処理
    huga_df = pl.read_csv('huga.csv', encoding='cp932')
    duplicate_groups = (huga_df
        .group_by('POL_ID')
        .agg([pl.col('*').alias('group_data')])
        .filter(pl.col('group_data').list.len() > 1)
    )
    
    if len(duplicate_groups) > 0:
        for col in huga_df.columns:
            if col != 'POL_ID':
                for group in duplicate_groups.iter_rows():
                    group_data = group['group_data']
                    values = [row[col] for row in group_data]
                    unique_values = list(set(values))
                    
                    if len(unique_values) > 1:
                        # 異なる値がある場合のみ、インデックス付き列を作成
                        for idx, value in enumerate(unique_values, 1):
                            huga_df = huga_df.with_columns([
                                pl.when(
                                    (pl.col('POL_ID') == group[0]) &
                                    (pl.col(col) == value)
                                )
                                .then(value)
                                .otherwise(None)
                                .alias(f'{col}_{idx}')
                            ])
    
    # 重複行を削除（POL_IDのみを基準とする）
    huga_df = huga_df.unique(subset=['POL_ID'])
    return hoge_df, huga_df, flg_columns

def categorize_content(df, cat_dict_path):
    """
    content1の内容を分類し、苦情フラグを追加する関数
    
    Args:
        df (pl.DataFrame): 分類対象のDataFrame
        cat_dict_path (str): カテゴリー辞書ファイルのパス
        
    Returns:
        pl.DataFrame: 分類済みのDataFrame
    """
    cat_dict = pl.read_csv(cat_dict_path, encoding='cp932')
    
    # 小分類と中分類の列を追加
    df = df.with_columns([
        pl.lit('others').alias('subcategory'),
        pl.lit('others').alias('category')
    ])
    
    # content1の小分類のマッチング
    for subcategory, category in zip(cat_dict['小分類'], cat_dict['中分類']):
        df = df.with_columns([
            pl.when(
                (pl.col('subcategory') == 'others') & 
                pl.col('content1').str.contains(subcategory)
            )
            .then(subcategory)
            .otherwise(pl.col('subcategory'))
            .alias('subcategory'),
            
            pl.when(
                (pl.col('category') == 'others') & 
                pl.col('content1').str.contains(subcategory)
            )
            .then(category)
            .otherwise(pl.col('category'))
            .alias('category')
        ])
    
    # 苦情フラグの追加
    df = df.with_columns(
        pl.col('content2').str.contains('苦情').alias('is_complaint')
    )
    
    return df

def main():
    """
    メイン処理を実行する関数
    
    Returns:
        tuple: (merged_df, daily_stats)
    """
    print("Reading main data files...")
    main_df = read_csv_in_chunks('temp/*.csv')
    
    print("Reading auxiliary files...")
    hoge_df, huga_df, flg_columns = load_auxiliary_files()
    
    print("Merging dataframes...")
    # データの結合
    merged_df = main_df.join(
        hoge_df,
        left_on='converted_contract_no',
        right_on='POL_NO',
        how='left'
    )
    
    merged_df = merged_df.join(
        huga_df,
        left_on='converted_contract_no',
        right_on='POL_ID',
        how='left'
    )
    
    print("Categorizing content...")
    merged_df = categorize_content(merged_df, 'cat_dict.csv')
    
    # 結果の保存
    print("Saving results...")
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 日次統計の保存
    daily_stats = (merged_df
        .groupby(pl.col('event_datetime').cast(pl.Date))
        .agg([
            pl.col('is_complaint').sum().alias('total_complaints')
        ])
    )
    
    daily_stats.write_csv(f'{output_dir}/daily_stats.csv')
    
    print("Analysis complete!")
    return merged_df, daily_stats

if __name__ == "__main__":
    merged_df, daily_stats = main()
