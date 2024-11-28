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
    
    dfs = []
    for file in glob.glob(file_pattern):
        print(f"Processing file: {file}")
        df = pl.scan_csv(file, has_header=False, new_columns=columns)
        # 会社区分=Gのデータのみをフィルタリング
        df = df.filter(pl.col('company_type') == 'G').select([
            'event_datetime', 'sales_person', 'content1', 'content2',
            'converted_contract_no', 'sales_person_id', 'customer_id'
        ])
        dfs.append(df)
    
    # 重複行の処理
    combined_df = pl.concat(dfs).collect()
    
    # event_dateとconverted_contract_noが同じ行の処理
    duplicates = (combined_df.groupby(['event_datetime', 'converted_contract_no'])
                 .agg(pl.count().alias('count'))
                 .filter(pl.col('count') > 1))
    
    if len(duplicates) > 0:
        # 重複行の処理
        for col in combined_df.columns:
            if col not in ['event_datetime', 'converted_contract_no']:
                # 重複行で値が異なる場合、新しい列を作成
                different_values = (combined_df.groupby(['event_datetime', 'converted_contract_no'])
                                  .agg(pl.col(col).list().alias(f'{col}_list'))
                                  .with_columns(pl.col(f'{col}_list').list.unique().alias(f'{col}_unique')))
                
                # 値が異なる場合、インデックス付きの列を作成
                for idx, unique_values in enumerate(different_values[f'{col}_unique'], 1):
                    if len(unique_values) > 1:
                        combined_df = combined_df.with_columns(
                            pl.col(col).alias(f'{col}_{idx}')
                            .filter(pl.col('event_datetime').is_in(different_values['event_datetime']) & 
                                  pl.col('converted_contract_no').is_in(different_values['converted_contract_no']))
                        )
    
    return combined_df

def load_auxiliary_files():
    """
    補助ファイルを読み込む関数
    """
    hoge_df = pl.read_csv('hoge.csv')
    flg_columns = [col for col in hoge_df.columns if col.endswith('FLG')]
    print(f"Found FLG columns: {flg_columns}")
    
    # hugaファイルの処理
    huga_df = pl.read_csv('huga.csv')
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
    cat_dict = pl.read_csv(cat_dict_path)
    
    # 小分類のマッチング
    for subcategory in cat_dict['小分類'].unique():
        df = df.with_columns(
            pl.col('content1').str.contains(subcategory).alias(f'contains_{subcategory}')
        )
    
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
