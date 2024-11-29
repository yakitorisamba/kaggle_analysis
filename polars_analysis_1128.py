import polars as pl
import glob
import os

def read_csv_in_chunks(file_pattern):
    """
    CSVファイルを読み込む関数（遅延評価使用）
    """
    base_columns = [
        'event_datetime', 'seq_no', 'sales_person', 'data_datetime', 
        'content1', 'content2', 'company_type', 'contract_no', 
        'converted_contract_no', 'source_id', 'url1', 'url2', 
        'customer_id'
    ]
    needed_columns = [
        'event_datetime', 'sales_person', 'content1', 'content2',
        'converted_contract_no', 'sales_person_id', 'customer_id'
    ]
    
    # 遅延評価を使用してファイルをスキャン
    dfs = []
    for file in glob.glob(file_pattern):
        print(f"Processing file: {file}")
        # 遅延評価でファイルをスキャン
        df = pl.scan_csv(file, has_header=False, encoding='utf-8')
        
        # 列名を動的に設定
        actual_columns = base_columns + [f'col_{i}' for i in range(14, 14 + (len(df.columns) - len(base_columns)))]
        df = df.with_columns([pl.col('^.*$').map(lambda x: x).keep_name()])
        df = df.rename(dict(zip(df.columns, actual_columns[:len(df.columns)])))
        
        # 必要な列の存在確認と型変換を遅延評価で実行
        for col in needed_columns:
            if col not in df.columns:
                print(f"Missing column: {col} in {file}")
                continue
        
        # Gデータのフィルタリングと必要な列の選択を遅延評価で実行
        df = df.filter(pl.col('company_type') == 'G').select(needed_columns)
        
        # 重複チェックと列の追加を遅延評価で実行
        duplicates = (df
            .group_by(['event_datetime', 'converted_contract_no'])
            .agg([pl.col('*').count().alias('count')])
            .filter(pl.col('count') > 1))
        
        if duplicates.collect().height > 0:
            for col in needed_columns:
                if col not in ['event_datetime', 'converted_contract_no']:
                    different_values = (df
                        .group_by(['event_datetime', 'converted_contract_no'])
                        .agg(pl.col(col).alias('vals'))
                        .filter(pl.col('vals').list.len() > 1))
                    
                    # 値が異なる場合のみ列を追加
                    if different_values.collect().height > 0:
                        max_dups = different_values.select(
                            pl.col('vals').list.len().alias('len')
                        ).collect().max().row(0)[0]
                        
                        # 列の追加を遅延評価で実行
                        for i in range(1, max_dups + 1):
                            df = df.with_columns(
                                pl.when(
                                    pl.col('event_datetime').is_in(different_values.select('event_datetime')) & 
                                    pl.col('converted_contract_no').is_in(different_values.select('converted_contract_no'))
                                )
                                .then(pl.col(col))
                                .otherwise(None)
                                .alias(f'{col}_{i}')
                            )
        
        # 重複行を削除
        df = df.unique(subset=['event_datetime', 'converted_contract_no'])
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid dataframes to concatenate")
    
    # 最終的なデータフレームを作成
    return pl.concat(dfs).collect()

def load_auxiliary_files():
    """
    補助ファイルを読み込む関数
    """
    hoge_df = pl.scan_csv('hoge.csv', encoding='utf-8')
    flg_columns = [col for col in hoge_df.columns if col.endswith('FLG')]
    
    # POL_IDの重複チェックを遅延評価で実行
    huga_df = pl.scan_csv('huga.csv', encoding='utf-8')
    duplicates = (huga_df
        .group_by('POL_ID')
        .agg([pl.col('*').count().alias('count')])
        .filter(pl.col('count') > 1))
    
    if duplicates.collect().height > 0:
        for col in huga_df.columns:
            if col != 'POL_ID':
                different_values = (huga_df
                    .group_by('POL_ID')
                    .agg(pl.col(col).alias('vals'))
                    .filter(pl.col('vals').list.len() > 1))
                
                if different_values.collect().height > 0:
                    max_dups = different_values.select(
                        pl.col('vals').list.len().alias('len')
                    ).collect().max().row(0)[0]
                    
                    for i in range(1, max_dups + 1):
                        huga_df = huga_df.with_columns(
                            pl.when(pl.col('POL_ID').is_in(different_values.select('POL_ID')))
                            .then(pl.col(col))
                            .otherwise(None)
                            .alias(f'{col}_{i}')
                        )
    
    return hoge_df.collect(), huga_df.collect(), flg_columns

def categorize_content(df, cat_dict_path):
    """
    content1の内容を分類する関数
    """
    cat_dict = pl.scan_csv(cat_dict_path, encoding='utf-8')
    df = df.lazy()
    
    # デフォルトカテゴリを設定
    df = df.with_columns([
        pl.lit('others').alias('subcategory'),
        pl.lit('others').alias('category')
    ])
    
    # カテゴリの割り当てを遅延評価で実行
    cat_pairs = cat_dict.select(['小分類', '中分類']).collect().rows()
    for subcategory, category in cat_pairs:
        df = df.with_columns([
            pl.when(
                (pl.col('subcategory') == 'others') & pl.col('content1').str.contains(subcategory)
            ).then(subcategory).otherwise(pl.col('subcategory')).alias('subcategory'),
            pl.when(
                (pl.col('category') == 'others') & pl.col('content1').str.contains(subcategory)
            ).then(category).otherwise(pl.col('category')).alias('category')
        ])
    
    # 苦情フラグ
    df = df.with_columns(
        pl.col('content2').str.contains('苦情').alias('is_complaint')
    )
    
    return df.collect()

def main():
    print("Reading main data files...")
    main_df = read_csv_in_chunks('temp/*.csv')
    
    print("Reading auxiliary files...")
    hoge_df, huga_df, flg_columns = load_auxiliary_files()
    
    print("Merging dataframes...")
    merged_df = main_df.lazy().join(
        hoge_df.lazy(),
        left_on='converted_contract_no',
        right_on='POL_NO',
        how='left'
    ).join(
        huga_df.lazy(),
        left_on='converted_contract_no',
        right_on='POL_ID',
        how='left'
    ).collect()
    
    print("Categorizing content...")
    final_df = categorize_content(merged_df, 'cat_dict.csv')
    
    print("Saving results...")
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    daily_stats = (final_df.lazy()
        .group_by(pl.col('event_datetime').cast(pl.Date))
        .agg([pl.col('is_complaint').sum().alias('total_complaints')])
        .collect()
    )
    
    daily_stats.write_csv(f'{output_dir}/daily_stats.csv')
    
    print("Analysis complete!")
    return final_df, daily_stats

if __name__ == "__main__":
    final_df, daily_stats = main()
