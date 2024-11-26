import pandas as pd
import numpy as np

def merge_duplicate_rows_optimized(df, pid_column='pid'):
    """
    同じPID値を持つ行を比較し、値が異なる列については新しい列を作成してマージします。
    パフォーマンスを最適化したバージョン。
    
    Parameters:
    -----------
    df : pandas.DataFrame
        入力データフレーム
    pid_column : str, default='pid'
        PID値が格納されている列名
    
    Returns:
    --------
    pandas.DataFrame
        マージ後のデータフレーム
    """
    # PIDの重複回数を計算
    duplicate_counts = df[pid_column].value_counts()
    
    # 重複のないPIDを持つ行を抽出
    unique_rows = df[df[pid_column].isin(duplicate_counts[duplicate_counts == 1].index)]
    
    # 重複のあるPIDを持つ行を抽出
    duplicate_rows = df[df[pid_column].isin(duplicate_counts[duplicate_counts > 1].index)]
    
    if len(duplicate_rows) == 0:
        return df.copy()
    
    # 重複行の処理
    result_rows = []
    
    # より効率的なグループ処理
    for pid, group in duplicate_rows.groupby(pid_column):
        # 最初の行をベースにする
        base_row = group.iloc[0].to_dict()
        
        # 各列の一意な値を効率的に取得
        unique_values = group.nunique()
        
        # 値が異なる列を特定
        diff_columns = unique_values[unique_values > 1].index
        
        # PID列は除外
        diff_columns = diff_columns[diff_columns != pid_column]
        
        if len(diff_columns) > 0:
            # 異なる値を持つ列について新しい列を作成
            for col in diff_columns:
                values = group[col].unique()
                for i, val in enumerate(values[1:], 1):
                    base_row[f"{col}_{i}"] = val
        
        result_rows.append(base_row)
    
    # 結果を組み立てる
    result_df = pd.concat([
        unique_rows,
        pd.DataFrame(result_rows)
    ], ignore_index=True)
    
    return result_df

# パフォーマンス比較用の関数
def compare_performance(df, n_iterations=5):
    """
    元の実装と最適化版のパフォーマンスを比較します。
    
    Parameters:
    -----------
    df : pandas.DataFrame
        テスト用データフレーム
    n_iterations : int
        各関数を実行する回数
    """
    import time
    
    def measure_time(func, df):
        start_time = time.time()
        for _ in range(n_iterations):
            func(df)
        end_time = time.time()
        return (end_time - start_time) / n_iterations
    
    original_time = measure_time(merge_duplicate_rows, df)
    optimized_time = measure_time(merge_duplicate_rows_optimized, df)
    
    print(f"元の実装の平均実行時間: {original_time:.4f}秒")
    print(f"最適化版の平均実行時間: {optimized_time:.4f}秒")
    print(f"速度向上率: {(original_time/optimized_time):.2f}倍")

# 使用例とパフォーマンステスト
if __name__ == "__main__":
    # より大きなサンプルデータの作成
    np.random.seed(42)
    n_rows = 10000
    
    data = {
        'pid': np.repeat(range(n_rows // 2), 2),  # 各PIDが2回ずつ出現
        'name': np.random.choice(['John', 'Alice', 'Bob', 'Carol'], n_rows),
        'age': np.random.randint(20, 60, n_rows),
        'city': np.random.choice(['NY', 'LA', 'Chicago', 'Boston'], n_rows),
        'score': np.random.randint(0, 100, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # パフォーマンス比較の実行
    print("パフォーマンス比較:")
    compare_performance(df)
    
    # 結果の確認
    result = merge_duplicate_rows_optimized(df.head())
    print("\n最初の数行の処理結果:")
    print(result)
