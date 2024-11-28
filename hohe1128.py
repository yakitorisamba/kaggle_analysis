import pandas as pd
import os

def process_files(list_file_path, new_result_dir, existing_result_dir, output_dir):
    # リストファイルの読み込み
    list_df = pd.read_csv(list_file_path)
    list_df.columns = ['filename', 'column_name', 'is_needed']  # リストファイルの列名を設定
    
    # 必要な列のみのディクショナリを作成
    needed_columns = dict(zip(list_df['filename'], 
                            [list_df[list_df['filename'] == fname]['column_name'][list_df[list_df['filename'] == fname]['is_needed']].tolist() 
                             for fname in list_df['filename'].unique()]))
    
    # 新しい結果の処理
    for filename in needed_columns.keys():
        # 新しい結果ファイルの読み込み
        new_file_path = os.path.join(new_result_dir, filename)
        if os.path.exists(new_file_path):
            new_df = pd.read_csv(new_file_path)
            
            # 必要な列のみを選択
            new_df = new_df[needed_columns[filename]]
            
            # 既存の結果ファイルの読み込みと結合
            existing_file_path = os.path.join(existing_result_dir, filename)
            if os.path.exists(existing_file_path):
                existing_df = pd.read_csv(existing_file_path)
                
                # 列名が一致していることを確認
                if set(existing_df.columns) == set(new_df.columns):
                    # データの結合
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    
                    # 重複行の削除（必要な場合）
                    combined_df = combined_df.drop_duplicates()
                    
                    # 結果の保存
                    output_file_path = os.path.join(output_dir, filename)
                    combined_df.to_csv(output_file_path, index=False)
                    print(f"Processed and saved: {filename}")
                else:
                    print(f"Column mismatch in {filename}")
            else:
                print(f"Existing file not found: {filename}")
        else:
            print(f"New result file not found: {filename}")

# 使用例
if __name__ == "__main__":
    list_file_path = "list.csv"
    new_result_dir = "result"
    existing_result_dir = "existing_results"
    output_dir = "combined_results"
    
    # 出力ディレクトリの作成（存在しない場合）
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイルの処理実行
    process_files(list_file_path, new_result_dir, existing_result_dir, output_dir)
