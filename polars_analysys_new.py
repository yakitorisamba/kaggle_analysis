import polars as pl
import glob
import os
from datetime import datetime

def read_csv_in_chunks(file_pattern):
    """
    CSVファイルを読み込む関数
    """
    columns = [
        'event_datetime', 'seq_no', 'sales_person', 'data_datetime', 
        'content1', 'content2', 'company_type', 'contract_no', 
        'converted_contract_no', 'source_id', 'url1', 'url2', 
        'customer_id'] + [f'col_{i}' for i in range(14, 19)] + ['sales_person_id'] + [f'col_{i}' for i in range(20, 27)]
    
    all_files = glob.glob(file_pattern)
    if not all_files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    # 最初のファイルを読み込み
    print(f"Processing file: {all_files[0]}")
    combined_df = pl.read_csv(
        all_files[0], 
        has_header=False, 
        new_columns=columns, 
        encoding='cp932'
    ).filter(pl.col('company_type') == 'G').select([
        'event_datetime', 'sales_person', 'content1', 'content2',
        'converted_contract_no', 'sales_person_id', 'customer_id'
    ])
    
    # 残りのファイルを処理
    for file in all_files[1:]:
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
        
        # 同じevent_dateとconverted_contract_noを持つ行の処理
        combined_df = pl.concat([combined_df, df])
        
        # 重複行の処理
        for col in combined_df.columns:
            if col not in ['event_datetime', 'converted_contract_no']:
                # 重複する行のグループごとに異なる値を持つものを特定
                different_values = (combined_df
                    .group_by(['event_datetime', 'converted_contract_no'])
                    .agg([
                        pl.col(col).list().unique().alias('unique_vals')
                    ])
                    .filter(pl.col('unique_vals').list.len() > 1)
                )
                
                # 異なる値が存在する場合、インデックス付きの列を作成
                if len(different_values) > 0:
                    for i in [1, 2]:  # _1, _2 のみ作成
                        idx = i - 1  # リストのインデックス用
                        combined_df = combined_df.with_columns([
                            pl.when(pl.col('event_datetime').is_in(different_values['event_datetime']) & 
                                  pl.col('converted_contract_no').is_in(different_values['converted_contract_no']))
                            .then(pl.col(col))
                            .alias(f'{col}_{i}')
                        ])
    
    # 重複行を削除（インデックス付きの列は保持）
    combined_df = combined_df.unique(subset=['event_datetime', 'converted_contract_no'])
    
    return combined_df

def load_auxiliary_files():
    """
    補助ファイルを読み込む関数
    """
    hoge_df = pl.read_csv('hoge.csv', encoding='cp932')
    flg_columns = [col for col in hoge_df.columns if col.endswith('FLG')]
    print(f"Found FLG columns: {flg_columns}")
    
    # hugaファイルの処理
    huga_df = pl.read_csv('huga.csv', encoding='cp932')
    duplicates = huga_df.groupby('POL_ID').agg(pl.count().alias('count')).filter(pl.col('count') > 1)
    
    if len(duplicates) > 0:
        # 重複行の処理
        for col in huga_df.columns:
            if col != 'POL_ID':
                # 重複行で値が異なる場合、新しい列を作成
                different_values = (huga_df.groupby('POL_ID')
                                  .agg(pl.col(col).list().alias(f'{col}_list'))
                                  .with_columns(pl.col(f'{col}_list').list.unique().alias(f'{col}_unique')))
                
                # 値が異なる場合、インデックス付きの列を作成
                for idx, unique_values in enumerate(different_values[f'{col}_unique'], 1):
                    if len(unique_values) > 1:
                        huga_df = huga_df.with_columns(
                            pl.col(col).alias(f'{col}_{idx}')
                            .filter(pl.col('POL_ID').is_in(different_values['POL_ID']))
                        )
    
    return hoge_df, huga_df, flg_columns

def categorize_content(df, cat_dict_path):
    """
    content1の内容を分類する関数
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
                (pl.col('subcategory') == 'others') & pl.col('content1').str.contains(subcategory)
            ).then(subcategory).otherwise(pl.col('subcategory')).alias('subcategory'),
            pl.when(
                (pl.col('category') == 'others') & pl.col('content1').str.contains(subcategory)
            ).then(category).otherwise(pl.col('category')).alias('category')
        ])
    
    # 苦情フラグの追加
    df = df.with_columns(
        pl.col('content2').str.contains('苦情').alias('is_complaint')
    )
    
    return df

def main():
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
    daily_stats = (merged_df.groupby(pl.col('event_datetime').cast(pl.Date))
                  .agg([
                      pl.col('is_complaint').sum().alias('total_complaints')
                  ]))
    
    daily_stats.write_csv(f'{output_dir}/daily_stats.csv')
    
    print("Analysis complete!")
    return merged_df, daily_stats

if __name__ == "__main__":
    merged_df, daily_stats = main()
