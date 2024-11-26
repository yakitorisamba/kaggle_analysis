import pandas as pd
import numpy as np
from typing import List, Set
import sys

def process_large_csv(
    file_path: str,
    chunk_size: int = 100000,
    key_columns: List[str] = None,
    memory_limit_gb: float = 4.0
) -> pd.DataFrame:
    """
    大規模CSVファイルを効率的に処理し、指定したキー列の値が同じ連続した行をマージします。
    
    Parameters:
    -----------
    file_path : str
        処理するCSVファイルのパス
    chunk_size : int
        一度に読み込むチャンクのサイズ
    key_columns : List[str]
        比較するキーとなる列名のリスト
    memory_limit_gb : float
        使用する最大メモリ制限（GB）
    
    Returns:
    --------
    pd.DataFrame
        処理結果のデータフレーム
    """
    
    # メモリ使用量を監視する関数
    def check_memory_usage():
        memory_usage = sys.getsizeof(result_df) / (1024 ** 3)  # GB単位
        if memory_usage > memory_limit_gb:
            raise MemoryError(f"メモリ使用量が制限値({memory_limit_gb}GB)を超えました")
    
    # 連続する行をマージする関数
    def merge_consecutive_rows(group_df: pd.DataFrame) -> pd.DataFrame:
        if len(group_df) == 1:
            return group_df
            
        result_row = group_df.iloc[0].copy()
        
        # 各列を比較し、異なる値がある場合は新しい列を作成
        for col in group_df.columns:
            if col in key_columns:
                continue
                
            values = group_df[col].unique()
            if len(values) > 1:
                for i, val in enumerate(values, 1):
                    result_row[f"{col}_{i}"] = val
                    
        return pd.DataFrame([result_row])
    
    # 結果を格納するデータフレーム
    result_df = pd.DataFrame()
    
    # CSVファイルを少しずつ読み込んで処理
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # キー列でソート
        chunk_sorted = chunk.sort_values(by=key_columns)
        
        # 連続する行をグループ化
        chunk_grouped = chunk_sorted.groupby(
            (chunk_sorted[key_columns] != chunk_sorted[key_columns].shift()).any(axis=1).cumsum()
        )
        
        # 各グループを処理
        processed_chunk = pd.concat([
            merge_consecutive_rows(group) 
            for _, group in chunk_grouped
        ])
        
        # 結果を追加
        result_df = pd.concat([result_df, processed_chunk], ignore_index=True)
        
        # メモリ使用量をチェック
        check_memory_usage()
    
    # 最終的な結果をキー列でソートして返す
    return result_df.sort_values(by=key_columns).reset_index(drop=True)

# 使用例
if __name__ == "__main__":
    # ファイルパスと設定
    file_path = "large_data.csv"
    key_columns = ["id", "date"]  # キーとなる列名を指定
    
    try:
        # データ処理の実行
        result = process_large_csv(
            file_path=file_path,
            chunk_size=100000,
            key_columns=key_columns,
            memory_limit_gb=4.0
        )
        
        # 結果の保存
        result.to_csv("processed_data.csv", index=False)
        print("処理が完了しました")
        
    except MemoryError as e:
        print(f"エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
